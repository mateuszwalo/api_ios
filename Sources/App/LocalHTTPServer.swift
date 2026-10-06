import Foundation
import Network

/// HTTP/1.1 server on the local network.
///
/// Three requirements shape it, and all three are unusual enough that they are easier to
/// meet directly than to configure out of a framework:
///
///   * **No timeouts anywhere.** A single response can take twenty minutes. Anything that
///     gives up on an idle socket ends the client's whole run, because the client does not
///     retry a transport error.
///   * **TCP keepalive on.** Twenty minutes of silence on an established connection is
///     exactly what a home router's NAT table garbage-collects. Keepalive probes make the
///     connection visibly alive while nothing is being sent.
///   * **A disconnect must be noticed while a request is in flight.** Otherwise a client
///     that went away still occupies the one inference slot, and everything queued behind
///     it waits for an answer nobody will read.
final class LocalHTTPServer: @unchecked Sendable {

    typealias Handler = @Sendable (HTTPRequestMessage) async -> HTTPResponseMessage

    enum State: Equatable {
        case stopped
        case starting
        case running(port: UInt16)
        case failed(String)
    }

    private let queue = DispatchQueue(label: "llm.http", qos: .userInitiated)
    private var listener: NWListener?
    private let handler: Handler
    private let onStateChange: @Sendable (State) -> Void

    init(handler: @escaping Handler, onStateChange: @escaping @Sendable (State) -> Void) {
        self.handler = handler
        self.onStateChange = onStateChange
    }

    func start(port: UInt16, advertiseBonjour: Bool = true) {
        stop()
        onStateChange(.starting)

        let options = NWProtocolTCP.Options()
        options.enableKeepalive = true
        // Probe well inside the window in which consumer NAT tables drop idle mappings.
        options.keepaliveIdle = 30
        options.keepaliveInterval = 15
        options.keepaliveCount = 8
        options.noDelay = true
        // No `connectionTimeout` or `connectionDropTime`: the defaults are generous and any
        // value set here would be a deadline on a request that is allowed to take as long
        // as it takes.

        let parameters = NWParameters(tls: nil, tcp: options)
        parameters.allowLocalEndpointReuse = true
        parameters.includePeerToPeer = false

        do {
            // Port 0 typed into the settings field would make this unwrap trap and take the
            // app down; reported as a failure to start instead.
            guard let endpointPort = NWEndpoint.Port(rawValue: port), port > 0 else {
                onStateChange(.failed("invalid port \(port)"))
                return
            }
            let listener = try NWListener(using: parameters, on: endpointPort)
            if advertiseBonjour {
                // Advertising is also what reliably prompts for local network permission on
                // iOS; without the prompt, the listener accepts nothing and says nothing.
                listener.service = NWListener.Service(name: "LocalLLM Server", type: "_http._tcp")
            }
            listener.stateUpdateHandler = { [weak self] state in
                guard let self else { return }
                switch state {
                case .ready:
                    self.onStateChange(.running(port: listener.port?.rawValue ?? port))
                case .failed(let error):
                    self.onStateChange(.failed(error.localizedDescription))
                case .cancelled:
                    self.onStateChange(.stopped)
                default:
                    break
                }
            }
            listener.newConnectionHandler = { [weak self] connection in
                guard let self else { return }
                let session = HTTPSession(connection: connection, queue: self.queue, handler: self.handler)
                session.start()
            }
            listener.start(queue: queue)
            self.listener = listener
        } catch {
            onStateChange(.failed(error.localizedDescription))
        }
    }

    func stop() {
        listener?.cancel()
        listener = nil
    }
}

/// One client connection, serving requests in order until the peer goes away.
private final class HTTPSession: @unchecked Sendable {

    private let connection: NWConnection
    private let queue: DispatchQueue
    private let handler: LocalHTTPServer.Handler
    private var parser = HTTPRequestParser()
    private var closed = false

    init(connection: NWConnection, queue: DispatchQueue, handler: @escaping LocalHTTPServer.Handler) {
        self.connection = connection
        self.queue = queue
        self.handler = handler
    }

    func start() {
        connection.start(queue: queue)
        Task.detached(priority: .userInitiated) { [self] in
            await serve()
            close()
        }
    }

    private func serve() async {
        while !closed {
            switch parser.next() {
            case .failure(let status, let message):
                let body = (try? JSONEncoder().encode(OpenAIErrorBody.invalidRequest(message))) ?? Data()
                _ = try? await send(HTTPResponseMessage.json(status, body).serialized(keepAlive: false))
                return

            case .complete(let request):
                let keepAlive = HTTPRequestParser.keepAlive(request)
                let response = await runHandler(for: request)
                // 499 is a note to the operator, not something to put on the wire: the peer
                // that would read it is the one that left.
                if response.status != 499 {
                    do { try await send(response.serialized(keepAlive: keepAlive)) }
                    catch { return }
                }
                if !keepAlive { return }

            case .incomplete:
                do {
                    guard let chunk = try await nextChunk() else { return }
                    parser.feed(chunk)
                } catch {
                    return
                }
            }
        }
    }

    /// A read started while a request was being handled, still in flight when it finished.
    ///
    /// There is never more than one receive outstanding on the connection, and only `serve`
    /// touches the parser. An earlier version let a watchdog task feed the parser while
    /// `serve` was reading it — a data race on a value type, invisible with curl, which opens
    /// one connection per request, and live with the OpenAI client, which keeps connections
    /// open and sends request after request down them.
    private var pendingRead: Task<Data?, Error>?

    /// The next bytes from the peer, continuing a read already in flight rather than issuing
    /// a second one beside it.
    private func nextChunk() async throws -> Data? {
        if let read = pendingRead {
            pendingRead = nil
            return try await read.value
        }
        return try await receive()
    }

    /// Runs the handler while watching the socket, so that a client hanging up mid-request
    /// cancels the work instead of leaving it to finish into the void.
    ///
    /// The watch is a single read that only reports. If the peer closes while the work runs,
    /// the work is cancelled; if bytes arrive instead — a pipelined request — they wait in the
    /// read's result until `serve` asks for them, and it is `serve` that feeds them in.
    private func runHandler(for request: HTTPRequestMessage) async -> HTTPResponseMessage {
        let work = Task { await handler(request) }

        let read = Task { [self] () throws -> Data? in
            do {
                let chunk = try await receive()
                if chunk == nil { work.cancel() }   // clean EOF: the client is gone
                return chunk
            } catch {
                work.cancel()
                throw error
            }
        }
        pendingRead = read

        return await work.value
    }

    private func receive() async throws -> Data? {
        try await withCheckedThrowingContinuation { continuation in
            connection.receive(minimumIncompleteLength: 1, maximumLength: 1 << 16) { data, _, isComplete, error in
                if let error {
                    continuation.resume(throwing: error)
                } else if let data, !data.isEmpty {
                    continuation.resume(returning: data)
                } else if isComplete {
                    continuation.resume(returning: nil)   // peer closed
                } else {
                    continuation.resume(returning: Data())
                }
            }
        }
    }

    private func send(_ data: Data) async throws {
        try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
            connection.send(content: data, completion: .contentProcessed { error in
                if let error { continuation.resume(throwing: error) }
                else { continuation.resume() }
            })
        }
    }

    private func close() {
        closed = true
        connection.cancel()
    }
}

// MARK: - Addresses

enum NetworkInterfaces {

    struct Address: Identifiable, Sendable, Equatable {
        let name: String        // en0, en1, …
        let ip: String
        var id: String { name + ip }
    }

    /// Every IPv4 address the device currently answers on.
    ///
    /// All of them, not just Wi-Fi: an iPad driven over a USB-C Ethernet adapter answers on
    /// a different interface, and guessing wrong leaves the operator typing an address that
    /// times out with no indication why.
    static func localIPv4() -> [Address] {
        var addresses: [Address] = []
        var head: UnsafeMutablePointer<ifaddrs>?
        guard getifaddrs(&head) == 0, let first = head else { return [] }
        defer { freeifaddrs(head) }

        for pointer in sequence(first: first, next: { $0.pointee.ifa_next }) {
            let flags = Int32(pointer.pointee.ifa_flags)
            guard flags & IFF_UP == IFF_UP, flags & IFF_LOOPBACK == 0 else { continue }
            guard let addr = pointer.pointee.ifa_addr, addr.pointee.sa_family == UInt8(AF_INET) else { continue }

            var host = [CChar](repeating: 0, count: Int(NI_MAXHOST))
            let result = getnameinfo(addr, socklen_t(addr.pointee.sa_len),
                                     &host, socklen_t(host.count),
                                     nil, 0, NI_NUMERICHOST)
            guard result == 0 else { continue }
            let ip = String(cString: host)
            let name = String(cString: pointer.pointee.ifa_name)
            if !ip.isEmpty && ip != "0.0.0.0" {
                addresses.append(Address(name: name, ip: ip))
            }
        }
        return addresses
    }
}
