package com.example.qwen3_tts.inference.runners

import com.example.qwen3_tts.config.QwenConfigLoader
import com.example.qwen3_tts.data.repository.*
import com.example.qwen3_tts.domain.builders.*

class QwenGenerateCodesRunner(
    private val tag: String = "Qwen3TTS"
) {
    data class GenerateCodesResult(
        val codes: LongArray,
        val timeSteps: Int,
        val codebooks: Int,
        val generatedGroup0Tokens: List<Int>
    )

    fun run(
        repo: QwenModelRepository,
        config: QwenConfigLoader.FullConfig,
        tokenIds: List<Int>,
        embeddings: QwenEmbeddingRepository,
        language: String = "auto",
        speakerId: Int = -1,
        maxNewTokens: Int = 64,
        temperature: Float = 1.0f,
        topK: Int = 50,
        repetitionPenalty: Float = 1.1f,
        minNewTokens: Int = 0
    ): GenerateCodesResult {
        require(tokenIds.isNotEmpty()) { "tokenIds must not be empty" }
        require(maxNewTokens > 0) { "maxNewTokens must be > 0" }

        val prefillBuilder = QwenPrefillBuilder(tag)
        val talkerPrefillRunner = QwenTalkerPrefillRunner(tag)
        val talkerDecodeRunner = QwenTalkerDecodeRunner(tag)
        val codePredictorLoopRunner = QwenCodePredictorLoopRunner(tag)
        val nextInputBuilder = QwenNextInputBuilder(tag)

        val prefill = prefillBuilder.build(
            tokenIds = tokenIds,
            config = config,
            embeddings = embeddings,
            language = language,
            speakerId = speakerId
        )

        val talkerPrefillFile = repo.getModelFile("talker_prefill.onnx")
        val talkerDecodeFile = repo.getModelFile("talker_decode.onnx")
        val codePredictorFile = repo.getModelFile("code_predictor.onnx")

        val prefillRunResult = talkerPrefillRunner.run(
            modelFile = talkerPrefillFile,
            prefill = prefill
        )

        val generatedSteps = mutableListOf<List<Int>>()
        val generatedGroup0Tokens = mutableListOf<Int>()

        val timestep0 = codePredictorLoopRunner.runFromPrefill(
            modelFile = codePredictorFile,
            config = config,
            prefillResult = prefillRunResult,
            embeddings = embeddings,
            previousGroup0Tokens = generatedGroup0Tokens,
            temperature = temperature,
            topK = topK,
            repetitionPenalty = repetitionPenalty,
            minNewTokens = minNewTokens,
            step = 0
        )

        if (timestep0.hitEos) {
            return GenerateCodesResult(
                codes = LongArray(0),
                timeSteps = 0,
                codebooks = 16,
                generatedGroup0Tokens = emptyList()
            )
        }

        generatedSteps.add(timestep0.codes16)
        generatedGroup0Tokens.add(timestep0.codes16[0])

        var nextInput = nextInputBuilder.buildFromCodes(
            codes16 = timestep0.codes16,
            timestepIndex = 0,
            prefill = prefill,
            config = config,
            embeddings = embeddings
        )

        var decodeRunResult = talkerDecodeRunner.runOneStepAfterPrefill(
            modelFile = talkerDecodeFile,
            config = config,
            prefillResult = prefillRunResult,
            nextInputEmbeds = nextInput
        )

        for (step in 1 until maxNewTokens) {
            val stepCodes = codePredictorLoopRunner.runFromDecode(
                modelFile = codePredictorFile,
                config = config,
                decodeResult = decodeRunResult,
                embeddings = embeddings,
                previousGroup0Tokens = generatedGroup0Tokens,
                temperature = temperature,
                topK = topK,
                repetitionPenalty = repetitionPenalty,
                minNewTokens = minNewTokens,
                step = step
            )

            if (stepCodes.hitEos) {
                break
            }

            generatedSteps.add(stepCodes.codes16)
            generatedGroup0Tokens.add(stepCodes.codes16[0])

            if (step == maxNewTokens - 1) {
                break
            }

            nextInput = nextInputBuilder.buildFromCodes(
                codes16 = stepCodes.codes16,
                timestepIndex = step,
                prefill = prefill,
                config = config,
                embeddings = embeddings
            )

            decodeRunResult = talkerDecodeRunner.runOneStepFromNextInput(
                modelFile = talkerDecodeFile,
                config = config,
                previousDecodeResult = decodeRunResult,
                nextInputEmbeds = nextInput
            )
        }

        val timeSteps = generatedSteps.size
        val flatCodes = flattenCodesToB16T(generatedSteps)

        return GenerateCodesResult(
            codes = flatCodes,
            timeSteps = timeSteps,
            codebooks = 16,
            generatedGroup0Tokens = generatedGroup0Tokens.toList()
        )
    }

    private fun flattenCodesToB16T(steps: List<List<Int>>): LongArray {
        if (steps.isEmpty()) return LongArray(0)

        val timeSteps = steps.size
        val codebooks = 16
        val out = LongArray(codebooks * timeSteps)

        for (t in 0 until timeSteps) {
            val step = steps[t]
            require(step.size == codebooks) {
                "Step $t has ${step.size} codebooks, expected $codebooks"
            }

            for (g in 0 until codebooks) {
                val idx = g * timeSteps + t
                out[idx] = step[g].toLong()
            }
        }

        return out
    }
}