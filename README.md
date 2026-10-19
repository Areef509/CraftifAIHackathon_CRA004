# AI Workplace Safety Sentinel
### Intelligent Workplace Safety Monitoring with PipeGen

## Overview
AI Workplace Safety Sentinel is a computer-vision-based safety monitoring system built with PipeGen to help identify workplace hazards before they escalate. It analyzes live camera feeds or recorded videos to monitor workers, verify safety compliance, and generate actionable incident alerts.

## Key Capabilities
- **Worker Detection & Tracking:** Detect people and maintain tracking IDs across frames.
- **Smart PPE Compliance:** Check safety helmets, reflective vests, goggles, gloves, protective footwear, and respirators where compatible detection models are available.
- **Restricted-Zone Intelligence:** Detect unauthorized entry, track zone exits, measure dwell time, and identify repeated entries.
- **Incident Management:** Generate timestamped JSON/JSONL records with event type, severity, confidence, worker ID, and available evidence.
- **Real-Time Alerting:** Publish safety events through MQTT for downstream monitoring and notifications.
- **Visual Monitoring:** Display annotated video and save incident images or clips where implemented.
- **Performance Analytics:** Measure FPS, inference latency, resource utilization, and video-processing stability.

## Intelligent Workflow

**Camera / Test Video → PipeGen Video Pipeline → Person & PPE Detection → Worker Tracking → Zone and Safety Rule Evaluation → Incident Classification → JSON Logging + MQTT Alerts + Evidence Capture → Performance Analysis**

The workflow combines computer vision with rule-based safety analytics to turn video observations into structured, actionable safety events.

## Technology Stack
**PipeGen · Computer Vision · Object Detection · Multi-Object Tracking · Python · JSON/JSONL · MQTT**

## Project Goal
Move workplace safety from reactive incident reporting toward proactive, AI-assisted monitoring through continuous visual analysis, traceable incident records, and timely alerts.

*Implementation note: Each capability is validated individually. PPE coverage depends on compatible models, and performance or alert-delivery claims are reported only after testing.*

**GitHub:** https://github.com/Areef509/ai-workplace-safety-sentinel
