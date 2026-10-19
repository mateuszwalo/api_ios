import Foundation

/// Which build of the app this is, as stamped in by CI.
///
/// Measurements are compared across builds, and a build changes results whenever it changes
/// how prompts are rendered or how schemas are enforced. So every response, the health check,
/// the stats and the self-test all carry the same identifier, and a row in a results table can
/// always be traced to the code that produced it.
///
/// The marketing version is set in project.yml and raised by hand with each release. The build
/// number is the CI run number and the commit is the git SHA, both injected at build time; a
/// local build shows "dev" instead, so it cannot be mistaken for a released one.
enum AppVersion {

    static var marketing: String { info("CFBundleShortVersionString") ?? "?" }
    static var buildNumber: String { info("CFBundleVersion") ?? "?" }
    static var commit: String { info("LLMBuildCommit").flatMap { $0.isEmpty ? nil : $0 } ?? "dev" }

    /// "1.2 (build 41, a1b2c3d)" — what to quote as the device build.
    static var description: String { "\(marketing) (build \(buildNumber), \(commit))" }

    private static func info(_ key: String) -> String? {
        Bundle.main.object(forInfoDictionaryKey: key) as? String
    }
}
