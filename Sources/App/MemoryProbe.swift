import Foundation
import LlamaBridge

/// Memory and thermal readings.
///
/// Footprint rather than resident size: the memory killer on iOS accounts for footprint, so
/// that is the number a budget has to be judged against. Resident size is reported too,
/// because it is what external tooling shows and the two diverging is itself informative.
enum MemoryProbe {

    static func footprintBytes() -> UInt64 { LLMBridge.physicalFootprintBytes() }

    static func availableBytes() -> UInt64 { LLMBridge.availableMemoryBytes() }

    static func thermalStateName() -> String {
        switch ProcessInfo.processInfo.thermalState {
        case .nominal:  return "nominal"
        case .fair:     return "fair"
        case .serious:  return "serious"
        case .critical: return "critical"
        @unknown default: return "unknown"
        }
    }

    static func format(_ bytes: UInt64) -> String {
        ByteCountFormatter.string(fromByteCount: Int64(bytes), countStyle: .memory)
    }

    /// True when the process is close enough to its limit that loading or generating is
    /// likely to be killed rather than fail.
    ///
    /// A guess by necessity — the limit is not published and depends on the device and on
    /// whether the increased-memory-limit entitlement survived signing. Half a gigabyte of
    /// headroom is enough to finish a response and report a problem, which is the whole
    /// point: the alternative is the app disappearing mid-run with no message anywhere.
    static func isUnderPressure() -> Bool {
        let available = availableBytes()
        return available > 0 && available < 512 * 1024 * 1024
    }
}
