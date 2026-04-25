package com.example.qwen3_tts.inference.runners

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.TensorInfo
import java.io.File
import java.nio.FloatBuffer
import java.nio.LongBuffer
import com.example.qwen3_tts.config.QwenConfigLoader

class QwenTalkerDecodeRunner(
    private val tag: String = "Qwen3TTS"
) {
    data class DecodeRunResult(
        val logits: FloatArray,
        val logitsShape: LongArray?,
        val hiddenStates: FloatArray,
        val hiddenStatesShape: LongArray?,
        val presentKeys: FloatArray,
        val presentKeysShape: LongArray?,
        val presentValues: FloatArray,
        val presentValuesShape: LongArray?
    )

    fun runOneStepFromNextInput(
        session: OrtSession,
        config: QwenConfigLoader.FullConfig,
        previousDecodeResult: DecodeRunResult,
        nextInputEmbeds: FloatArray
    ): DecodeRunResult {
        val env = OrtEnvironment.getEnvironment()

        val hiddenSize = config.talker.hiddenSize
        val numLayers = config.talker.numHiddenLayers
        val numKvHeads = config.talker.numKeyValueHeads
        val headDim = config.talker.headDim

        val previousPresentKeysShape = requireNotNull(previousDecodeResult.presentKeysShape) {
            "previousDecodeResult.presentKeysShape is null"
        }
        val pastSeqLen = previousPresentKeysShape[3].toInt()
        val totalSeqLen = pastSeqLen + 1

        val attentionMask = LongArray(totalSeqLen) { 1L }
        val positionIds = longArrayOf(
            pastSeqLen.toLong(),
            pastSeqLen.toLong(),
            pastSeqLen.toLong()
        )

        val pastKeysShape = longArrayOf(
            numLayers.toLong(),
            1L,
            numKvHeads.toLong(),
            pastSeqLen.toLong(),
            headDim.toLong()
        )

        val pastValuesShape = longArrayOf(
            numLayers.toLong(),
            1L,
            numKvHeads.toLong(),
            pastSeqLen.toLong(),
            headDim.toLong()
        )

        val inputsEmbedsTensor = OnnxTensor.createTensor(
            env,
            FloatBuffer.wrap(nextInputEmbeds),
            longArrayOf(1, 1, hiddenSize.toLong())
        )

        val attentionMaskTensor = OnnxTensor.createTensor(
            env,
            LongBuffer.wrap(attentionMask),
            longArrayOf(1, totalSeqLen.toLong())
        )

        val positionIdsTensor = OnnxTensor.createTensor(
            env,
            LongBuffer.wrap(positionIds),
            longArrayOf(3, 1, 1)
        )

        val pastKeysTensor = OnnxTensor.createTensor(
            env,
            FloatBuffer.wrap(previousDecodeResult.presentKeys),
            pastKeysShape
        )

        val pastValuesTensor = OnnxTensor.createTensor(
            env,
            FloatBuffer.wrap(previousDecodeResult.presentValues),
            pastValuesShape
        )

        inputsEmbedsTensor.use { embeds ->
            attentionMaskTensor.use { mask ->
                positionIdsTensor.use { pos ->
                    pastKeysTensor.use { pk ->
                        pastValuesTensor.use { pv ->
                            val inputs = mapOf(
                                "inputs_embeds" to embeds,
                                "attention_mask" to mask,
                                "position_ids" to pos,
                                "past_keys" to pk,
                                "past_values" to pv
                            )

                            session.run(inputs).use { outputs ->
                                return readDecodeOutputs(outputs, "next-input")
                            }
                        }
                    }
                }
            }
        }
    }

    fun runOneStepAfterPrefill(
        session: OrtSession,
        config: QwenConfigLoader.FullConfig,
        prefillResult: QwenTalkerPrefillRunner.PrefillRunResult,
        nextInputEmbeds: FloatArray
    ): DecodeRunResult {
        val env = OrtEnvironment.getEnvironment()
        val hiddenSize = config.talker.hiddenSize
        val numLayers = config.talker.numHiddenLayers
        val numKvHeads = config.talker.numKeyValueHeads
        val headDim = config.talker.headDim

        require(nextInputEmbeds.size == hiddenSize) {
            "nextInputEmbeds size must be $hiddenSize, got ${nextInputEmbeds.size}"
        }

        val prefillKeysShape = requireNotNull(prefillResult.presentKeysShape) {
            "prefill presentKeysShape is null"
        }
        val pastSeqLen = prefillKeysShape[3].toInt()
        val totalSeqLen = pastSeqLen + 1
        val attentionMask = LongArray(totalSeqLen) { 1L }

        val positionIds = longArrayOf(
            pastSeqLen.toLong(),
            pastSeqLen.toLong(),
            pastSeqLen.toLong()
        )

        val pastKeysShape = longArrayOf(
            numLayers.toLong(),
            1L,
            numKvHeads.toLong(),
            pastSeqLen.toLong(),
            headDim.toLong()
        )

        val pastValuesShape = longArrayOf(
            numLayers.toLong(),
            1L,
            numKvHeads.toLong(),
            pastSeqLen.toLong(),
            headDim.toLong()
        )

        val inputsEmbedsTensor = OnnxTensor.createTensor(
            env,
            FloatBuffer.wrap(nextInputEmbeds),
            longArrayOf(1, 1, hiddenSize.toLong())
        )

        val attentionMaskTensor = OnnxTensor.createTensor(
            env,
            LongBuffer.wrap(attentionMask),
            longArrayOf(1, totalSeqLen.toLong())
        )

        val positionIdsTensor = OnnxTensor.createTensor(
            env,
            LongBuffer.wrap(positionIds),
            longArrayOf(3, 1, 1)
        )

        val pastKeysTensor = OnnxTensor.createTensor(
            env,
            FloatBuffer.wrap(prefillResult.presentKeys),
            pastKeysShape
        )

        val pastValuesTensor = OnnxTensor.createTensor(
            env,
            FloatBuffer.wrap(prefillResult.presentValues),
            pastValuesShape
        )

        inputsEmbedsTensor.use { embeds ->
            attentionMaskTensor.use { mask ->
                positionIdsTensor.use { pos ->
                    pastKeysTensor.use { pk ->
                        pastValuesTensor.use { pv ->
                            val inputs = mapOf(
                                "inputs_embeds" to embeds,
                                "attention_mask" to mask,
                                "position_ids" to pos,
                                "past_keys" to pk,
                                "past_values" to pv
                            )

                            session.run(inputs).use { outputs ->
                                return readDecodeOutputs(outputs, "after-prefill")
                            }
                        }
                    }
                }
            }
        }
    }

    fun runOneStepAfterPrefill(
        modelFile: File,
        config: QwenConfigLoader.FullConfig,
        prefillResult: QwenTalkerPrefillRunner.PrefillRunResult,
        nextInputEmbeds: FloatArray
    ): DecodeRunResult {
        require(modelFile.exists()) { "talker_decode model does not exist: ${modelFile.absolutePath}" }
        require(modelFile.length() > 0L) { "talker_decode model is empty: ${modelFile.absolutePath}" }

        val env = OrtEnvironment.getEnvironment()
        val sessionOptions = OrtSession.SessionOptions()
        val session = env.createSession(modelFile.absolutePath, sessionOptions)

        try {
            return runOneStepAfterPrefill(session, config, prefillResult, nextInputEmbeds)
        } finally {
            session.close()
            sessionOptions.close()
        }
    }

    fun runOneStepFromNextInput(
        modelFile: File,
        config: QwenConfigLoader.FullConfig,
        previousDecodeResult: DecodeRunResult,
        nextInputEmbeds: FloatArray
    ): DecodeRunResult {
        require(modelFile.exists()) { "talker_decode model does not exist: ${modelFile.absolutePath}" }
        require(modelFile.length() > 0L) { "talker_decode model is empty: ${modelFile.absolutePath}" }

        val env = OrtEnvironment.getEnvironment()
        val sessionOptions = OrtSession.SessionOptions()
        val session = env.createSession(modelFile.absolutePath, sessionOptions)

        try {
            return runOneStepFromNextInput(session, config, previousDecodeResult, nextInputEmbeds)
        } finally {
            session.close()
            sessionOptions.close()
        }
    }

    private fun readDecodeOutputs(
        outputs: OrtSession.Result,
        phase: String
    ): DecodeRunResult {
        val logitsTensor =
            outputs.firstOrNull { it.key == "logits" }?.value as? OnnxTensor
                ?: error("Missing decode logits")
        val hiddenStatesTensor =
            outputs.firstOrNull { it.key == "hidden_states" }?.value as? OnnxTensor
                ?: error("Missing decode hidden_states")
        val presentKeysTensor =
            outputs.firstOrNull { it.key == "present_keys" }?.value as? OnnxTensor
                ?: error("Missing decode present_keys")
        val presentValuesTensor =
            outputs.firstOrNull { it.key == "present_values" }?.value as? OnnxTensor
                ?: error("Missing decode present_values")

        val logitsShape = (logitsTensor.info as? TensorInfo)?.shape
        val hiddenStatesShape = (hiddenStatesTensor.info as? TensorInfo)?.shape
        val presentKeysShapeOut = (presentKeysTensor.info as? TensorInfo)?.shape
        val presentValuesShapeOut = (presentValuesTensor.info as? TensorInfo)?.shape

        val logits = logitsTensor.floatBuffer.let {
            val arr = FloatArray(it.remaining())
            it.get(arr)
            arr
        }

        val hiddenStates = hiddenStatesTensor.floatBuffer.let {
            val arr = FloatArray(it.remaining())
            it.get(arr)
            arr
        }

        val presentKeys = presentKeysTensor.floatBuffer.let {
            val arr = FloatArray(it.remaining())
            it.get(arr)
            arr
        }

        val presentValues = presentValuesTensor.floatBuffer.let {
            val arr = FloatArray(it.remaining())
            it.get(arr)
            arr
        }

        return DecodeRunResult(
            logits = logits,
            logitsShape = logitsShape,
            hiddenStates = hiddenStates,
            hiddenStatesShape = hiddenStatesShape,
            presentKeys = presentKeys,
            presentKeysShape = presentKeysShapeOut,
            presentValues = presentValues,
            presentValuesShape = presentValuesShapeOut
        )
    }
}