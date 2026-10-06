import { Route, Routes } from "react-router-dom";

import { Layout } from "./components/Layout";
import { ModeProvider } from "./lib/mode";
import { DashboardPage } from "./pages/DashboardPage";
import { RunPage } from "./pages/RunPage";

export default function App() {
  return (
    <ModeProvider>
      <Layout>
        <Routes>
          <Route path="/" element={<DashboardPage />} />
          <Route path="/runs/:researchId" element={<RunPage />} />
          <Route path="*" element={<p className="py-24 text-center text-muted">Page not found.</p>} />
        </Routes>
      </Layout>
    </ModeProvider>
  );
}
