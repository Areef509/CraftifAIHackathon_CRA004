// Custom detection (bbox) parser (generated).
// Scaffold validates the output layer and exposes the raw tensor(s); the DECODE
// block (per-model) populates `objectList` with NvDsInferObjectDetectionInfo.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "nvdsinfer_custom_impl.h"

extern "C" bool ParseConfigPersonDet(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferObjectDetectionInfo>& objectList);

CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(ParseConfigPersonDet);

extern "C" bool ParseConfigPersonDet(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferObjectDetectionInfo>& objectList)
{
    if (outputLayersInfo.empty()) return false;
    auto const& layer0 = outputLayersInfo[0];

    const unsigned numClasses = detectionParams.numClassesConfigured;
    const float netW = static_cast<float>(networkInfo.width);
    const float netH = static_cast<float>(networkInfo.height);
    const float* data = static_cast<const float*>(layer0.buffer);

    static bool logged = false;
    if (!logged) {
        logged = true;
        fprintf(stderr, "[PersonDet] first call: layer=%s numDims=%d "
                "networkInput=%.0fx%.0f numClasses=%u outputs=%zu\n",
                layer0.layerName ? layer0.layerName : "?",
                layer0.inferDims.numDims, netW, netH, numClasses,
                outputLayersInfo.size());
    }

    // ==== BEGIN GENERATED DECODE ====
    // Output tensor shape (batch stripped): [84, 8400]
    // layer0.inferDims: numDims=2, d[0]=84, d[1]=8400
    // Layout: 84 rows x N columns. First 4 rows = cx, cy, w, h (in input pixel
    // coords). Rows 4..83 = class scores (post-sigmoid probabilities).
    // We need to transpose logically: for each of the N proposals, read across
    // the 84 rows at column index j.

    const unsigned int featureDim = layer0.inferDims.d[0]; // 84
    const unsigned int numProposals = layer0.inferDims.d[1]; // 8400

    // Sanity: featureDim should be 4 + numClasses
    if (featureDim != 4u + numClasses) {
        fprintf(stderr, "[PersonDet] WARNING: featureDim=%u != 4+numClasses=%u\n",
                featureDim, 4u + numClasses);
    }

    for (unsigned int j = 0; j < numProposals; ++j) {
        // Find best class score for this proposal
        float maxScore = 0.0f;
        unsigned int bestClassId = 0;
        for (unsigned int c = 0; c < numClasses; ++c) {
            float score = data[(4u + c) * numProposals + j];
            if (score > maxScore) {
                maxScore = score;
                bestClassId = c;
            }
        }

        // Per-class threshold
        float thr = (bestClassId < detectionParams.perClassPreclusterThreshold.size())
                     ? detectionParams.perClassPreclusterThreshold[bestClassId]
                     : 0.35f;

        if (maxScore < thr) continue;

        // Read box: cx, cy, w, h in network-input pixel coords
        float cx = data[0u * numProposals + j];
        float cy = data[1u * numProposals + j];
        float bw = data[2u * numProposals + j];
        float bh = data[3u * numProposals + j];

        // Convert centre+size → top-left corner + size
        float left = cx - bw * 0.5f;
        float top  = cy - bh * 0.5f;

        // Clamp to network input
        left = std::max(0.0f, std::min(left, netW));
        top  = std::max(0.0f, std::min(top,  netH));
        bw   = std::max(0.0f, std::min(bw, netW - left));
        bh   = std::max(0.0f, std::min(bh, netH - top));

        // Skip degenerate boxes
        if (bw <= 1.0f || bh <= 1.0f) continue;

        NvDsInferObjectDetectionInfo obj;
        obj.classId = bestClassId;
        obj.left = left;
        obj.top = top;
        obj.width = bw;
        obj.height = bh;
        obj.detectionConfidence = maxScore;
        objectList.push_back(obj);
    }
    // ==== END GENERATED DECODE ====
    return true;
}