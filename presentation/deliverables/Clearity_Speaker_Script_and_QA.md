# Clearity: five-minute English speaker script

Target pace: approximately 140 words per minute. Allow short pauses and a brief demo transition.

## Slide 1

Hello, we are presenting Clearity, a prototype for reviewing silent cashier videos. Our starting point is simple: a reviewer needs to understand what happened, where it happened in the video, and what evidence supports a concern. Clearity organizes that review around a binary screening result, timestamps and selected frames. The system runs video inference inside its own service, without sending frames to an external AI provider. We also avoid face identification. This is an experimental decision-support tool. A human reviewer remains responsible for interpreting the evidence and deciding what further investigation is appropriate.

## Slide 2

Manual review requires sustained attention, especially when an action is brief or partly hidden. Silent footage adds another difficulty: the video cannot explain an instruction, an authorization or the accounting context behind a transaction. The challenge gives us MOV videos of roughly twenty-nine megabytes, lasting from ten to one hundred and forty-five seconds. The required processing limit is five minutes. Our goal is to make review easier to navigate by returning evidence tied to time. We do not claim measured labor savings, and we cannot infer intent or establish a legal violation from appearance alone.

## Slide 3

This is the required product journey. A user uploads a MOV video through the authenticated API, then checks the asynchronous job status. The completed result provides a binary prediction and event timestamps. For each event, the intended review package contains one to three JPEG evidence frames, a plain-language explanation and a regulatory reference. The reviewer can then examine the original context. We have implemented the jury API contract, including the analyze endpoint and the job endpoint. An event means a candidate for review. Because the baseline learns only binary labels, we cannot claim validated semantic violation categories or precise action boundaries.

## Slide 4

The architecture separates the deployable baseline from a more demanding experiment. The hosted configuration uses FastAPI, asynchronous jobs and SQLite, with OpenCV sampling frames every two seconds. A frozen ResNet18 encoder produces visual features. Temporal features feed a trained logistic-regression classifier, and the service assembles the response and evidence. All inference happens inside that server. Separately, our local Qwen3-VL experiment runs a quantized two-billion-parameter model through llama.cpp and inspects three frames per twenty-second window. Qwen is not part of the free hosted pipeline because of memory limits. Authentication uses X-API-Key, and the service deletes uploaded video copies after processing.

## Slide 5

The dataset is small, so we report the result directly. We have forty-two labeled records representing forty unique videos after exact-duplicate grouping. The saved experiment uses twenty-four unique videos for training, eight for validation and eight for the held-out test. Test accuracy, precision and recall are each fifty percent, with an F1 score of zero point five. That does not establish useful production detection. It gives us a reproducible baseline to improve. For regulatory context, we are reviewing Kazakhstan National Bank Resolution number two hundred thirty-one from November twenty-ninth, twenty nineteen. We provide general context, while verified event-to-clause mapping remains unfinished.

## Slide 6

For the demonstration, we show an upload, the processing state and the review result. These storyboard placeholders are conceptual and contain no banking footage. The repository and service links on the slide are confirmed. The Render deployment is live at clearity-3qoy.onrender.com. A hosted end-to-end test completed in sixty-three-point-six seconds, including upload, processing and polling. The input was a synthetic MOV of twenty-nine mebibytes, lasting one hundred forty-five seconds. This single run used a warm service. It verifies the API workflow within five minutes for that sample, while detection quality and cold-start timing remain unproven. Free Render hosting may sleep after inactivity.

## Slide 7

Our next work follows the limitations we have measured. First, expand independent labeled data and evaluate across camera angles. Second, replace coarse binary screening with event annotations and better temporal localization. Third, ask a qualified domain reviewer to verify the regulatory mapping. Finally, benchmark the deployed service, including cold starts, retention and failure handling. Sampling can miss brief actions, and generalization to a different camera has not been established. Clearity already demonstrates the structure of an accountable review workflow. The result we want to build toward is evidence that a reviewer can inspect, question and use responsibly. Thank you.

# Jury Q&A

**Why is accuracy only 50%?**
The held-out test contains only eight unique videos. The baseline does not demonstrate useful production detection. We report that result honestly and prioritize independent data, event annotations and stronger evaluation.

**Does video leave the service for AI inference?**
No. The hosted baseline runs OpenCV, ResNet18 and logistic regression inside its own server. The optional Qwen experiment runs locally through llama.cpp. No external AI inference API receives video frames. Hosted inference is server-side, so uploaded videos do travel to our deployed server.

**How do you connect events to Kazakhstan law?**
We reference National Bank Resolution No. 231 of 29 November 2019 as general cash-operations context: https://adilet.zan.kz/rus/docs/V1900019680. A qualified reviewer still needs to verify event-to-clause mapping. A visual prediction does not establish a legal violation.

**How do you handle privacy?**
We use X-API-Key authentication, avoid face identification, and delete uploaded server-side video copies after processing. Training videos, annotations, API keys and private results are excluded from GitHub. This prototype has not undergone a comprehensive privacy or security audit.

**Will it work from a different camera angle?**
That has not been established. The next evaluation should use independently collected views, locations and subjects. Frame sampling can also miss very brief actions.

**Is the hosting reliable and within five minutes?**
A warm hosted end-to-end run completed upload, processing and polling in 63.59 seconds for a synthetic 145.0-second, 29 MiB MOV. This verifies the five-minute limit for that one synthetic sample. It does not validate real-video detection quality, cold-start timing or uptime. Free Render hosting can sleep after inactivity.

**What does a positive result mean?**
The model finds visual similarity to labeled concerning examples. It does not establish theft, intent, authorization, an accounting error or a specific cash-discipline violation. A human must review the evidence and transaction context.

**What is Qwen contributing?**
It is an optional experimental mode for visual observations, using Qwen3-VL-2B-Instruct in quantized GGUF form through llama.cpp. The free hosted pipeline uses the smaller ResNet baseline because Qwen does not fit the server memory budget.

**Are the demo images real?**
The deck uses conceptual storyboard placeholders and editable diagrams. It includes no confidential banking video frames and no invented screenshots.
