import React from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import DigitalMind from './components/DigitalMind';
import BrainDashboard from './components/brain/BrainDashboard';

const App: React.FC = () => {
  return (
    <BrowserRouter>
      <Routes>
        {/* NIVEK Brain Dashboard */}
        <Route path="/brain/*" element={<BrainDashboard />} />

        {/* Original DigitalMind at root */}
        <Route path="/" element={
          <div className="w-full h-screen bg-black text-white">
            <DigitalMind />
          </div>
        } />

        {/* Redirect unknown routes to root */}
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </BrowserRouter>
  );
};

export default App;
