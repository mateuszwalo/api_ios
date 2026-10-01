import Foundation

/// One download in progress or recently finished.
struct DownloadJob: Identifiable, Sendable, Equatable {
    enum State: Sendable, Equatable {
        case waiting
        case running(received: Int64, expected: Int64)
        case paused(received: Int64, expected: Int64)
        case finished(URL)
        case failed(String)
    }

    let id: UUID
    let url: URL
    let destinationName: String
    var state: State

    var fractionComplete: Double {
        switch state {
        case .running(let received, let expected), .paused(let received, let expected):
            return expected > 0 ? Double(received) / Double(expected) : 0
        case .finished: return 1
        default: return 0
        }
    }

    var statusText: String {
        switch state {
        case .waiting: return "waiting"
        case .running(let received, let expected):
            return "\(ByteCountFormatter.string(fromByteCount: received, countStyle: .file)) of \(ByteCountFormatter.string(fromByteCount: expected, countStyle: .file))"
        case .paused(let received, _):
            return "paused at \(ByteCountFormatter.string(fromByteCount: received, countStyle: .file))"
        case .finished: return "done"
        case .failed(let why): return why
        }
    }
}

/// Downloads model files, surviving the network dropping out.
///
/// Three gigabytes over a shared network takes long enough that an interruption is the
/// expected case, not the exceptional one. Two mechanisms cover it: the session waits for
/// connectivity instead of failing when it is absent, and an interrupted transfer keeps its
/// resume data so it continues from where it stopped rather than from zero.
///
/// A foreground session, deliberately. A background one would survive the app being
/// suspended, but the server only runs in the foreground anyway, so the complexity buys
/// nothing here.
@MainActor
@Observable
final class ModelDownloader: NSObject {

    private(set) var jobs: [DownloadJob] = []

    /// For gated repositories. Held in memory only: a token written to disk on a sideloaded
    /// device that is passed around a lab is a liability, and re-entering it costs seconds.
    var huggingFaceToken: String = ""

    private var session: URLSession!
    private var tasks: [Int: UUID] = [:]            // URLSessionTask.taskIdentifier → job
    private var resumeData: [UUID: Data] = [:]
    private let destinationDirectory: URL

    init(destinationDirectory: URL) {
        self.destinationDirectory = destinationDirectory
        super.init()
        let configuration = URLSessionConfiguration.default
        configuration.waitsForConnectivity = true
        // No resource timeout: a slow network must stall the transfer, never cancel it.
        configuration.timeoutIntervalForRequest = 300
        configuration.timeoutIntervalForResource = .infinity
        configuration.allowsExpensiveNetworkAccess = true
        configuration.allowsConstrainedNetworkAccess = true
        session = URLSession(configuration: configuration, delegate: self, delegateQueue: nil)
    }

    // MARK: Control

    func enqueue(url: URL, as name: String? = nil) {
        let destinationName = name ?? url.lastPathComponent
        let job = DownloadJob(id: UUID(), url: url, destinationName: destinationName, state: .waiting)
        jobs.append(job)
        startTask(for: job, resuming: nil)
    }

    func enqueue(_ entry: ModelCatalog.Entry) {
        enqueue(url: entry.modelURL)
        if let projector = entry.projectorURL { enqueue(url: projector) }
    }

    func pause(_ jobID: UUID) {
        guard let identifier = tasks.first(where: { $0.value == jobID })?.key else { return }
        session.getAllTasks { tasks in
            guard let task = tasks.first(where: { $0.taskIdentifier == identifier }) as? URLSessionDownloadTask else { return }
            task.cancel { data in
                Task { @MainActor [weak self] in
                    guard let self else { return }
                    if let data { self.resumeData[jobID] = data }
                    self.update(jobID) { job in
                        if case .running(let received, let expected) = job.state {
                            job.state = .paused(received: received, expected: expected)
                        }
                    }
                }
            }
        }
    }

    func resume(_ jobID: UUID) {
        guard let job = jobs.first(where: { $0.id == jobID }) else { return }
        startTask(for: job, resuming: resumeData[jobID])
    }

    func cancel(_ jobID: UUID) {
        pause(jobID)
        jobs.removeAll { $0.id == jobID }
        resumeData[jobID] = nil
    }

    func clearFinished() {
        jobs.removeAll {
            if case .finished = $0.state { return true }
            if case .failed = $0.state { return true }
            return false
        }
    }

    // MARK: Plumbing

    private func startTask(for job: DownloadJob, resuming data: Data?) {
        let task: URLSessionDownloadTask
        if let data {
            task = session.downloadTask(withResumeData: data)
        } else {
            var request = URLRequest(url: job.url)
            if !huggingFaceToken.isEmpty {
                request.setValue("Bearer \(huggingFaceToken)", forHTTPHeaderField: "Authorization")
            }
            task = session.downloadTask(with: request)
        }
        tasks[task.taskIdentifier] = job.id
        update(job.id) { $0.state = .running(received: 0, expected: 0) }
        task.resume()
    }

    private func update(_ jobID: UUID, _ change: (inout DownloadJob) -> Void) {
        guard let index = jobs.firstIndex(where: { $0.id == jobID }) else { return }
        change(&jobs[index])
    }

    fileprivate func jobID(for task: URLSessionTask) -> UUID? { tasks[task.taskIdentifier] }

    fileprivate func finish(_ jobID: UUID, movingFrom location: URL, suggested: String) {
        let destination = destinationDirectory.appendingPathComponent(suggested)
        do {
            try? FileManager.default.createDirectory(at: destinationDirectory, withIntermediateDirectories: true)
            if FileManager.default.fileExists(atPath: destination.path) {
                try FileManager.default.removeItem(at: destination)
            }
            try FileManager.default.moveItem(at: location, to: destination)
            update(jobID) { $0.state = .finished(destination) }
        } catch {
            update(jobID) { $0.state = .failed("could not save: \(error.localizedDescription)") }
        }
    }

    fileprivate func fail(_ jobID: UUID, _ message: String, resume data: Data?) {
        if let data { resumeData[jobID] = data }
        update(jobID) { job in
            // A failure that carries resume data is an interruption, not a dead end: it is
            // reported as paused so the obvious next action is to continue it.
            if data != nil, case .running(let received, let expected) = job.state {
                job.state = .paused(received: received, expected: expected)
            } else {
                job.state = .failed(message)
            }
        }
    }

    fileprivate func progress(_ jobID: UUID, received: Int64, expected: Int64) {
        update(jobID) { $0.state = .running(received: received, expected: expected) }
    }
}

extension ModelDownloader: URLSessionDownloadDelegate {

    nonisolated func urlSession(_ session: URLSession,
                                downloadTask: URLSessionDownloadTask,
                                didFinishDownloadingTo location: URL) {
        // The temporary file is deleted as soon as this returns, so it is moved here and
        // not on a later hop to the main actor.
        let moved = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString + ".gguf")
        try? FileManager.default.moveItem(at: location, to: moved)

        let suggested = downloadTask.response?.suggestedFilename
            ?? downloadTask.originalRequest?.url?.lastPathComponent
            ?? "model.gguf"

        Task { @MainActor [weak self] in
            guard let self, let id = self.jobID(for: downloadTask) else { return }
            if let http = downloadTask.response as? HTTPURLResponse, http.statusCode >= 400 {
                try? FileManager.default.removeItem(at: moved)
                self.fail(id, "server returned \(http.statusCode)" +
                          (http.statusCode == 401 || http.statusCode == 403
                           ? " — this repository may need a Hugging Face token" : ""),
                          resume: nil)
                return
            }
            self.finish(id, movingFrom: moved, suggested: suggested)
        }
    }

    nonisolated func urlSession(_ session: URLSession,
                                downloadTask: URLSessionDownloadTask,
                                didWriteData bytesWritten: Int64,
                                totalBytesWritten: Int64,
                                totalBytesExpectedToWrite: Int64) {
        Task { @MainActor [weak self] in
            guard let self, let id = self.jobID(for: downloadTask) else { return }
            self.progress(id, received: totalBytesWritten, expected: totalBytesExpectedToWrite)
        }
    }

    nonisolated func urlSession(_ session: URLSession,
                                task: URLSessionTask,
                                didCompleteWithError error: Error?) {
        guard let error else { return }
        let resume = (error as NSError).userInfo[NSURLSessionDownloadTaskResumeData] as? Data
        Task { @MainActor [weak self] in
            guard let self, let id = self.jobID(for: task) else { return }
            self.fail(id, error.localizedDescription, resume: resume)
        }
    }
}
