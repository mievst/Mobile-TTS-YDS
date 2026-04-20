package com.example.qwen3_tts.data.repository

import java.io.Closeable
import java.io.File
import kotlin.math.sqrt
import com.example.qwen3_tts.data.npy.NpyReader
import com.example.qwen3_tts.data.npy.NpyFloatRowReader

class QwenEmbeddingRepository(
    private val modelRepository: QwenModelRepository,
    private val tag: String = "Qwen3TTS"
) : Closeable {

    private val npyReader = NpyReader()

    private var textEmbeddingRowReader: NpyFloatRowReader? = null
    private var talkerCodecEmbeddingRowReader: NpyFloatRowReader? = null
    private val cpCodecEmbeddingReaders = mutableMapOf<Int, NpyFloatRowReader>()

    private var textProjectionFc1WeightCache: NpyReader.NpyFloatArray? = null
    private var textProjectionFc1BiasCache: NpyReader.NpyFloatArray? = null
    private var textProjectionFc2WeightCache: NpyReader.NpyFloatArray? = null
    private var textProjectionFc2BiasCache: NpyReader.NpyFloatArray? = null

    fun loadRequiredCoreEmbeddings() {
        textEmbeddingRowReader = NpyFloatRowReader(modelRepository.getEmbeddingFile("text_embedding.npy"))
        talkerCodecEmbeddingRowReader = NpyFloatRowReader(modelRepository.getEmbeddingFile("talker_codec_embedding.npy"))

        for (i in 0 until 15) {
            cpCodecEmbeddingReaders[i] =
                NpyFloatRowReader(modelRepository.getEmbeddingFile("cp_codec_embedding_$i.npy"))
        }

        textProjectionFc1WeightCache = loadSmallEmbeddingFile("text_projection_fc1_weight.npy")
        textProjectionFc1BiasCache = loadSmallEmbeddingFile("text_projection_fc1_bias.npy")
        textProjectionFc2WeightCache = loadSmallEmbeddingFile("text_projection_fc2_weight.npy")
        textProjectionFc2BiasCache = loadSmallEmbeddingFile("text_projection_fc2_bias.npy")
    }

    fun textEmbeddingRow(tokenId: Int): FloatArray {
        val reader = requireNotNull(textEmbeddingRowReader) { "text_embedding.npy row reader not loaded" }
        return reader.readRow(tokenId)
    }

    fun talkerCodecEmbeddingRow(tokenId: Int): FloatArray {
        val reader = requireNotNull(talkerCodecEmbeddingRowReader) { "talker_codec_embedding.npy row reader not loaded" }
        return reader.readRow(tokenId)
    }

    fun cpCodecEmbeddingRow(groupIndex: Int, tokenId: Int): FloatArray {
        val reader = cpCodecEmbeddingReaders[groupIndex]
            ?: error("cp_codec_embedding_$groupIndex.npy reader not loaded")
        return reader.readRow(tokenId)
    }

    fun textProjection(input2048: FloatArray): FloatArray {
        val fc1W = requireNotNull(textProjectionFc1WeightCache) { "fc1 weight not loaded" }
        val fc1B = requireNotNull(textProjectionFc1BiasCache) { "fc1 bias not loaded" }
        val fc2W = requireNotNull(textProjectionFc2WeightCache) { "fc2 weight not loaded" }
        val fc2B = requireNotNull(textProjectionFc2BiasCache) { "fc2 bias not loaded" }

        require(fc1W.shape.contentEquals(intArrayOf(2048, 2048))) {
            "Unexpected fc1 weight shape: ${fc1W.shape.contentToString()}"
        }
        require(fc1B.shape.contentEquals(intArrayOf(2048))) {
            "Unexpected fc1 bias shape: ${fc1B.shape.contentToString()}"
        }
        require(fc2W.shape.contentEquals(intArrayOf(1024, 2048))) {
            "Unexpected fc2 weight shape: ${fc2W.shape.contentToString()}"
        }
        require(fc2B.shape.contentEquals(intArrayOf(1024))) {
            "Unexpected fc2 bias shape: ${fc2B.shape.contentToString()}"
        }
        require(input2048.size == 2048) {
            "textProjection expects input size 2048, got ${input2048.size}"
        }

        val fc1Out = FloatArray(2048)
        for (row in 0 until 2048) {
            var sum = fc1B.data[row].toDouble()
            val rowOffset = row * 2048
            for (col in 0 until 2048) {
                sum += input2048[col].toDouble() * fc1W.data[rowOffset + col].toDouble()
            }
            fc1Out[row] = GeLU(sum).toFloat()
        }

        val fc2Out = FloatArray(1024)
        for (row in 0 until 1024) {
            var sum = fc2B.data[row].toDouble()
            val rowOffset = row * 2048
            for (col in 0 until 2048) {
                sum += fc1Out[col].toDouble() * fc2W.data[rowOffset + col].toDouble()
            }
            fc2Out[row] = sum.toFloat()
        }

        return fc2Out
    }

    override fun close() {
        textEmbeddingRowReader?.close()
        talkerCodecEmbeddingRowReader?.close()
        cpCodecEmbeddingReaders.values.forEach { it.close() }

        textEmbeddingRowReader = null
        talkerCodecEmbeddingRowReader = null
        cpCodecEmbeddingReaders.clear()
    }

    private fun loadSmallEmbeddingFile(name: String): NpyReader.NpyFloatArray {
        val file: File = modelRepository.getEmbeddingFile(name)
        val arr = npyReader.readFloatArray(file)
        return arr
    }

    private fun GeLU(x: Double): Double {
        return 0.5 * x * (1.0 + kotlin.math.tanh(sqrt(2.0 / Math.PI) * (x + 0.044715 * x * x * x)))
    }
}