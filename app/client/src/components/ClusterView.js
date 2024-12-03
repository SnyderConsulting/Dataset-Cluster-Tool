import React from 'react';
import ImageColumn from './ImageColumn';

function ClusterView({ clusters, selectedImages, setSelectedImages }) {
  return (
    <div className="cluster-container">
      {clusters.map((cluster) => (
        <ImageColumn
          key={cluster.clusterName}
          cluster={cluster}
          selectedImages={selectedImages}
          setSelectedImages={setSelectedImages}
        />
      ))}
    </div>
  );
}

export default ClusterView;
