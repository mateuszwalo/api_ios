import SwiftUI

@main
struct LocalLLMServerApp: App {
    @State private var controller = ServerController()

    var body: some Scene {
        WindowGroup {
            RootView()
                .environment(controller)
        }
    }
}

struct RootView: View {
    @Environment(ServerController.self) private var controller

    var body: some View {
        TabView {
            ServerView()
                .tabItem { Label("Server", systemImage: "network") }
            ModelsView()
                .tabItem { Label("Models", systemImage: "shippingbox") }
            LogsView()
                .tabItem { Label("Requests", systemImage: "list.bullet.rectangle") }
        }
    }
}
