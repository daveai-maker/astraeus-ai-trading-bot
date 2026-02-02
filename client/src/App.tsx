import { Switch, Route } from "wouter";
import { queryClient } from "./lib/queryClient";
import { QueryClientProvider } from "@tanstack/react-query";
import { Toaster } from "@/components/ui/toaster";
import { useAuth } from "@/hooks/use-auth";
import NotFound from "@/pages/not-found";
import Overview from "@/pages/Overview";
import Trading from "@/pages/Trading";
import Backtester from "@/pages/Backtester";
import Logs from "@/pages/Logs";
import Settings from "@/pages/Settings";
import Forum from "@/pages/Forum";
import News from "@/pages/News";
import Manual from "@/pages/Manual";
import Landing from "@/pages/Landing";
import AdvancedOrders from "@/pages/AdvancedOrders";
import Sidebar from "@/components/layout/Sidebar";
import { Loader2 } from "lucide-react";

function LoadingScreen() {
  return (
    <div className="min-h-screen bg-black flex items-center justify-center">
      <div className="text-center">
        <Loader2 className="h-8 w-8 text-primary animate-spin mx-auto mb-4" />
        <p className="text-white/50 text-sm">Loading...</p>
      </div>
    </div>
  );
}

function AuthenticatedApp() {
  return (
    <Sidebar>
      <Switch>
        <Route path="/" component={Overview} />
        <Route path="/trading" component={Trading} />
        <Route path="/advanced-orders" component={AdvancedOrders} />
        <Route path="/backtester" component={Backtester} />
        <Route path="/logs" component={Logs} />
        <Route path="/news" component={News} />
        <Route path="/forum" component={Forum} />
        <Route path="/manual" component={Manual} />
        <Route path="/settings" component={Settings} />
        <Route component={NotFound} />
      </Switch>
    </Sidebar>
  );
}

function Router() {
  const { user, isLoading, isAuthenticated } = useAuth();

  if (isLoading) {
    return <LoadingScreen />;
  }

  if (!isAuthenticated) {
    return <Landing />;
  }

  return <AuthenticatedApp />;
}

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <Router />
      <Toaster />
    </QueryClientProvider>
  );
}

export default App;
