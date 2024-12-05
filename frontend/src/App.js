import React, { useState, useEffect } from 'react';
import axios from 'axios';
import ClusterView from './components/ClusterView';
import SelectionCounts from './components/SelectionCounts'; // New component
import './App.css';

function App() {
  const [clusters, setClusters] = useState([]);
  const [selectedImages, setSelectedImages] = useState({}); // Changed to an object

  useEffect(() => {
    axios.get('http://localhost:5050/api/clusters').then((response) => {
      let clusters = response.data;

      // Sort clusters numerically
      clusters.sort((a, b) => {
        const numA = parseInt(a.clusterName.replace('cluster_', ''), 10);
        const numB = parseInt(b.clusterName.replace('cluster_', ''), 10);
        return numA - numB;
      });

      setClusters(clusters);

      // Initialize selectedImages with empty arrays for each cluster
      const initialSelections = {};
      clusters.forEach((cluster) => {
        initialSelections[cluster.clusterName] = [];
      });
      setSelectedImages(initialSelections);
    });
  }, []);

  const handleExport = () => {
    // Flatten selectedImages to get an array of image paths
    const selectedImagePaths = [];
    for (const clusterName in selectedImages) {
      selectedImages[clusterName].forEach((imagePath) => {
        selectedImagePaths.push(imagePath);
      });
    }

    axios
      .post('http://localhost:5050/api/export', {
        selectedImages: selectedImagePaths,
      })
      .then((response) => {
        alert('Export complete!');
      })
      .catch((error) => {
        console.error('Error exporting images:', error);
        alert('Error exporting images.');
      });
  };

  // Compute total selected images
  const totalSelectedImages = Object.values(selectedImages).reduce(
    (sum, clusterSelections) => sum + clusterSelections.length,
    0
  );

  // Compute total images
  const totalImages = clusters.reduce(
    (sum, cluster) => sum + cluster.images.length,
    0
  );

  return (
    <div className="app-container">
      <header className="app-header">
        <h1>Image Cluster Viewer</h1>
      </header>
      <div className="app-content">
        {/* New component to display selection counts */}
        <SelectionCounts
          clusters={clusters}
          selectedImages={selectedImages}
          totalSelectedImages={totalSelectedImages}
          totalImages={totalImages}
        />
        <ClusterView
          clusters={clusters}
          selectedImages={selectedImages}
          setSelectedImages={setSelectedImages}
        />
      </div>
      <footer className="app-footer">
        <button onClick={handleExport}>Export</button>
      </footer>
    </div>
  );
}

export default App;