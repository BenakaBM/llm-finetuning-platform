import { BrowserRouter, Routes, Route } from "react-router-dom";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { AuthProvider } from "./hooks/useAuth";
import AppLayout from "./components/AppLayout";
import LoginPage from "./pages/LoginPage";
import RegisterPage from "./pages/RegisterPage";
import DashboardPage from "./pages/DashboardPage";
import DatasetsPage from "./pages/DatasetsPage";
import ModelsPage from "./pages/ModelsPage";
import FineTuningPage from "./pages/FineTuningPage";
import TrainingJobsPage from "./pages/TrainingJobsPage";
import TrainingJobDetailPage from "./pages/TrainingJobDetailPage";
import ExperimentsPage from "./pages/ExperimentsPage";
import ExperimentDetailPage from "./pages/ExperimentDetailPage";
import PlaygroundPage from "./pages/PlaygroundPage";

const queryClient = new QueryClient();

export default function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <AuthProvider>
        <BrowserRouter>
          <Routes>
            <Route path="/login" element={<LoginPage />} />
            <Route path="/register" element={<RegisterPage />} />
            <Route element={<AppLayout />}>
              <Route path="/" element={<DashboardPage />} />
              <Route path="/datasets" element={<DatasetsPage />} />
              <Route path="/models" element={<ModelsPage />} />
              <Route path="/fine-tuning" element={<FineTuningPage />} />
              <Route path="/training-jobs" element={<TrainingJobsPage />} />
              <Route path="/training-jobs/:jobId" element={<TrainingJobDetailPage />} />
              <Route path="/experiments" element={<ExperimentsPage />} />
              <Route path="/experiments/:experimentId" element={<ExperimentDetailPage />} />
              <Route path="/playground" element={<PlaygroundPage />} />
            </Route>
          </Routes>
        </BrowserRouter>
      </AuthProvider>
    </QueryClientProvider>
  );
}
