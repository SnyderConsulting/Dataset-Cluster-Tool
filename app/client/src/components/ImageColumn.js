import React from 'react';
import ImageItem from './ImageItem';

function ImageColumn({ cluster, selectedImages, setSelectedImages }) {
  return (
    <div className="image-column">
      <h3>{cluster.clusterName}</h3>
      {cluster.images.map((image, index) => (
        <ImageItem
          key={index}
          image={image}
          selectedImages={selectedImages}
          setSelectedImages={setSelectedImages}
          clusterName={cluster.clusterName}
        />
      ))}
    </div>
  );
}

export default ImageColumn;
