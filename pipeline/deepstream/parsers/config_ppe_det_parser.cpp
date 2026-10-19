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

extern "C" bool ParseConfigPpeDet(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferObjectDetectionInfo>& objectList);

CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(ParseConfigPpeDet);

extern "C" bool ParseConfigPpeDet(
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
        fprintf(stderr, "[PPE_DET] first call: layer=%s numDims=%d "
                "networkInput=%.0fx%.0f numClasses=%u outputs=%zu\n",
                layer0.layerName ? layer0.layerName : "?",
                layer0.inferDims.numDims, netW, netH, numClasses,
                outputLayersInfo.size());
    }

    // ==== BEGIN GENERATED DECODE ====
    // Output tensor shape (batch stripped): [10, 8400]
    // inferDims: numDims=2, d[0]=10, d[1]=8400
    // Layout: rows = 4 box coords + 6 class scores = 10, cols = 8400 proposals
    // We need to transpose: for proposal j, attribute i is at data[i * numProposals + j]

    const unsigned int featureDim = layer0.inferDims.d[0]; // 10
    const unsigned int numProposals = layer0.inferDims.d[1]; // 8400

    // Sanity: featureDim should be 4 + numClasses
    if (featureDim != 4u + numClasses) {
        fprintf(stderr, "[PPE_DET] ERROR: featureDim=%u != 4+numClasses=%u\n",
                featureDim, 4u + numClasses);
        return false;
    }

    for (unsigned int j = 0; j < numProposals; ++j) {
        // Transposed access: row i, col j -> data[i * numProposals + j]
        const float cx = data[0u * numProposals + j];
        const float cy = data[1u * numProposals + j];
        const float bw = data[2u * numProposals + j];
        const float bh = data[3u * numProposals + j];

        // Find best class
        float maxScore = 0.0f;
        unsigned int bestClass = 0;
        for (unsigned int c = 0; c < numClasses; ++c) {
            float s = data[(4u + c) * numProposals + j];
            if (s > maxScore) {
                maxScore = s;
                bestClass = c;
            }
        }

        // Threshold (scores are already post-sigmoid from ultralytics YOLO export)
        float thr = (bestClass < detectionParams.perClassPreclusterThreshold.size())
                     ? detectionParams.perClassPreclusterThreshold[bestClass]
                     : 0.3f;
        if (maxScore < thr) continue;

        // Convert cx,cy,w,h (in network-input pixel coords) to left,top,width,height
        float left   = cx - bw * 0.5f;
        float top    = cy - bh * 0.5f;
        float right  = cx + bw * 0.5f;
        float bottom = cy + bh * 0.5f;

        // Clamp to network input
        left   = std::max(0.0f, std::min(left,   netW));
        top    = std::max(0.0f, std::min(top,    netH));
        right  = std::max(0.0f, std::min(right,  netW));
        bottom = std::max(0.0f, std::min(bottom, netH));

        float w = right - left;
        float h = bottom - top;
        if (w <= 1.0f || h <= 1.0f) continue;

        NvDsInferObjectDetectionInfo obj;
        obj.classId = bestClass;
        obj.left = left;
        obj.top = top;
        obj.width = w;
        obj.height = h;
        obj.detectionConfidence = maxScore;
        objectList.push_back(obj);
    }
    // ==== END GENERATED DECODE ====
    return true;
}