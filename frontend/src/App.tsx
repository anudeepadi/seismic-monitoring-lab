import { Routes, Route } from 'react-router-dom';
import { AnimatePresence } from 'framer-motion';
import Layout from './components/Layout';
import Dashboard from './pages/Dashboard';
import Training from './pages/Training';
import Visualization from './pages/Visualization';
import Models from './pages/Models';
import Experiments from './pages/Experiments';
import SeismicData from './pages/SeismicData';

function App() {
  return (
    <Layout>
      <AnimatePresence mode="wait">
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/training" element={<Training />} />
          <Route path="/visualization" element={<Visualization />} />
          <Route path="/models" element={<Models />} />
          <Route path="/experiments" element={<Experiments />} />
          <Route path="/seismic" element={<SeismicData />} />
        </Routes>
      </AnimatePresence>
    </Layout>
  );
}

export default App;
