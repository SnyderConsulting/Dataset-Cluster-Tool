import React, { useState } from 'react';
import Modal from 'react-modal';

Modal.setAppElement('#root');

function ImageItem({
  image,
  selectedImages,
  setSelectedImages,
  clusterName,
}) {
  const [isExpanded, setIsExpanded] = useState(false);

  const imagePath = `${clusterName}/${image.filename}`;

  const isSelected = selectedImages[clusterName].includes(imagePath);

  const toggleSelect = () => {
    const clusterSelections = selectedImages[clusterName];
    if (isSelected) {
      // Remove the image from the selection
      const newSelections = clusterSelections.filter((img) => img !== imagePath);
      setSelectedImages({
        ...selectedImages,
        [clusterName]: newSelections,
      });
    } else {
      // Add the image to the selection
      setSelectedImages({
        ...selectedImages,
        [clusterName]: [...clusterSelections, imagePath],
      });
    }
  };

  return (
    <div className="image-item">
      <img
        src={`http://localhost:5050${image.url}`}
        alt={image.filename}
        className={isSelected ? 'selected' : ''}
        onClick={toggleSelect}
      />
      {/* Expand Icon */}
      <button
        onClick={(e) => {
          e.stopPropagation(); // Prevent triggering toggleSelect
          setIsExpanded(true);
        }}
        className="expand-button"
        title="Expand"
      >
        🔍
      </button>

      {/* Modal for Expanded Image */}
      <Modal
        isOpen={isExpanded}
        onRequestClose={() => setIsExpanded(false)}
        contentLabel="Expanded Image"
        style={{
          content: {
            maxWidth: '80%',
            maxHeight: '80%',
            margin: 'auto',
          },
        }}
      >
        <img
          src={`http://localhost:5050${image.url}`}
          alt={image.filename}
          style={{ maxWidth: '100%', maxHeight: '80vh' }}
        />
        <button onClick={() => setIsExpanded(false)}>Close</button>
      </Modal>
    </div>
  );
}

export default ImageItem;