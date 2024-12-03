const express = require('express');
const path = require('path');
const fs = require('fs');
const cors = require('cors');

const app = express();
const PORT = 5050;

app.use(cors());

// Replace with your actual output directory path
const OUTPUT_DIR = path.join(__dirname, 'output');
const EXPORT_DIR = path.join(__dirname, 'exported_images');

// Endpoint to get cluster data
app.get('/api/clusters', (req, res) => {
  fs.readdir(OUTPUT_DIR, (err, clusters) => {
    if (err) {
      console.error(err);
      res.status(500).send('Error reading clusters');
      return;
    }

    // Filter and sort clusters numerically
    const clusterFolders = clusters
      .filter((cluster) => cluster.startsWith('cluster_'))
      .sort((a, b) => {
        const numA = parseInt(a.replace('cluster_', ''), 10);
        const numB = parseInt(b.replace('cluster_', ''), 10);
        return numA - numB;
      });

    const clusterData = clusterFolders.map((cluster) => {
      const clusterPath = path.join(OUTPUT_DIR, cluster);
      const images = fs
        .readdirSync(clusterPath)
        .filter((file) => /\.(jpg|jpeg|png|gif|bmp)$/i.test(file))
        .map((file) => ({
          filename: file,
          url: `/images/${cluster}/${file}`,
        }));
      return {
        clusterName: cluster,
        images,
      };
    });
    res.json(clusterData);
  });
});

// Serve images statically
app.use('/images', express.static(OUTPUT_DIR));

// Handle export
app.post('/api/export', express.json(), (req, res) => {
  const selectedImages = req.body.selectedImages; // Expecting an array of image paths

  if (!selectedImages || !Array.isArray(selectedImages)) {
    return res.status(400).send('Invalid data');
  }

  // Ensure export directory exists
  if (!fs.existsSync(EXPORT_DIR)) {
    fs.mkdirSync(EXPORT_DIR);
  }

  selectedImages.forEach((imgPath) => {
    const srcPath = path.join(OUTPUT_DIR, imgPath);
    const destPath = path.join(EXPORT_DIR, path.basename(imgPath));

    // Copy the file
    fs.copyFile(srcPath, destPath, (err) => {
      if (err) {
        console.error(`Error exporting image ${imgPath}:`, err);
      }
    });
  });

  res.send('Export complete');
});

app.listen(PORT, () => {
  console.log(`Server is running on port ${PORT}`);
});
