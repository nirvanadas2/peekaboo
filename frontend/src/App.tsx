import { BrowserRouter, Routes, Route } from "react-router-dom";
import LandingPage from "./pages/LandingPage";
import DashboardPlaceholder from "./pages/DashboardPlaceholder";
import "./App.css";

const App = () => {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<LandingPage />} />
        <Route path="/dashboard" element={<DashboardPlaceholder />} />
      </Routes>
    </BrowserRouter>
  );
};

export default App;
