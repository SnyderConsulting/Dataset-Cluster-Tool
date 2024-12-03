function SelectionCounts({
  clusters,
  selectedImages,
  totalSelectedImages,
  totalImages,
}) {
  // Compute overall selection percentage
  const selectionPercentage =
    totalImages > 0 ? (totalSelectedImages / totalImages) * 100 : 0;

  return (
    <div className="selection-counts-container">
      <div className="total-selection">
        <div className="total-selection-text">
          Total Selected Images: {totalSelectedImages} / {totalImages}
        </div>
        <div className="selection-count-bar overall-bar">
          <div
            className="selection-count-bar-selected overall-bar-selected"
            style={{
              width: `${selectionPercentage}%`,
            }}
          ></div>
        </div>
      </div>
      <div className="selection-counts">
        {clusters.map((cluster) => {
          const clusterName = cluster.clusterName;
          const totalClusterImages = cluster.images.length;
          const selectedCount =
            selectedImages[clusterName] ? selectedImages[clusterName].length : 0;

          return (
            <div key={clusterName} className="selection-count-item">
              <div className="selection-count-cluster-name">
                {clusterName}
              </div>
              <div className="selection-count-bar">
                <div
                  className="selection-count-bar-selected"
                  style={{
                    width: `${(selectedCount / totalClusterImages) * 100}%`,
                  }}
                ></div>
              </div>
              <div className="selection-count-text">
                {selectedCount} / {totalClusterImages}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

export default SelectionCounts;