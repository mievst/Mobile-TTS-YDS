package com.example.qwen3_tts.data.npy

import java.io.File
import java.io.RandomAccessFile
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.charset.Charset

class NpyReader {

    data class NpyFloatArray(
        val shape: IntArray,
        val data: FloatArray
    )

    fun readFloatArray(file: File): NpyFloatArray {
        val metadata = NpyFormat.readMetadataFromBytes(file)
        val bytes = file.readBytes()

        val elementCount = metadata.shape.fold(1) { acc, dim -> acc * dim }
        val expectedBytes = elementCount * 4

        require(metadata.dataOffset + expectedBytes <= bytes.size) {
            "NPY data truncated in ${file.name}: expected $expectedBytes bytes of data"
        }

        val floatData = FloatArray(elementCount)
        val buffer = ByteBuffer
            .wrap(bytes, metadata.dataOffset, expectedBytes)
            .order(ByteOrder.LITTLE_ENDIAN)

        for (i in 0 until elementCount) {
            floatData[i] = buffer.getFloat()
        }

        return NpyFloatArray(
            shape = metadata.shape,
            data = floatData
        )
    }
}

class NpyFloatRowReader(file: File) : AutoCloseable {

    val filePath: String = file.absolutePath
    val shape: IntArray
    val rows: Int
    val cols: Int

    private val raf: RandomAccessFile = RandomAccessFile(file, "r")
    private val dataOffset: Long

    init {
        val metadata = NpyFormat.readMetadataFromRaf(file, raf)

        require(metadata.shape.size == 2) {
            "NpyFloatRowReader expects rank-2 array, got ${metadata.shape.contentToString()} in ${file.name}"
        }

        shape = metadata.shape
        rows = shape[0]
        cols = shape[1]
        dataOffset = metadata.dataOffset.toLong()
    }

    fun readRow(rowIndex: Int): FloatArray {
        require(rowIndex in 0 until rows) {
            "Row index $rowIndex out of range [0, $rows)"
        }

        val rowByteCount = cols * 4
        val offset = dataOffset + rowIndex.toLong() * rowByteCount.toLong()

        val rowBytes = ByteArray(rowByteCount)
        raf.seek(offset)
        raf.readFully(rowBytes)

        val buffer = ByteBuffer.wrap(rowBytes).order(ByteOrder.LITTLE_ENDIAN)
        val result = FloatArray(cols)
        for (i in 0 until cols) {
            result[i] = buffer.getFloat()
        }
        return result
    }

    override fun close() {
        raf.close()
    }
}

private object NpyFormat {

    private val ASCII: Charset = Charsets.US_ASCII

    data class Metadata(
        val major: Int,
        val minor: Int,
        val shape: IntArray,
        val dataOffset: Int
    )

    fun readMetadataFromBytes(file: File): Metadata {
        require(file.exists()) { "NPY file does not exist: ${file.absolutePath}" }
        require(file.length() > 0L) { "NPY file is empty: ${file.absolutePath}" }

        val bytes = file.readBytes()
        require(bytes.size >= 16) { "Invalid NPY file (too small): ${file.absolutePath}" }

        validateMagic(bytes, file.name)

        val major = bytes[6].toInt() and 0xFF
        val minor = bytes[7].toInt() and 0xFF

        val headerLength: Int
        val headerStart: Int
        val dataOffset: Int

        when (major) {
            1 -> {
                headerLength = (bytes[8].toInt() and 0xFF) or
                        ((bytes[9].toInt() and 0xFF) shl 8)
                headerStart = 10
                dataOffset = headerStart + headerLength
            }
            2, 3 -> {
                headerLength = (bytes[8].toInt() and 0xFF) or
                        ((bytes[9].toInt() and 0xFF) shl 8) or
                        ((bytes[10].toInt() and 0xFF) shl 16) or
                        ((bytes[11].toInt() and 0xFF) shl 24)
                headerStart = 12
                dataOffset = headerStart + headerLength
            }
            else -> {
                throw IllegalArgumentException("Unsupported NPY version: $major.$minor")
            }
        }

        require(dataOffset <= bytes.size) {
            "Invalid NPY header offset in ${file.name}"
        }

        val header = bytes
            .copyOfRange(headerStart, dataOffset)
            .toString(ASCII)

        return parseHeader(file.name, major, minor, header, dataOffset)
    }

    fun readMetadataFromRaf(file: File, raf: RandomAccessFile): Metadata {
        require(file.exists()) { "NPY file does not exist: ${file.absolutePath}" }
        require(file.length() > 0L) { "NPY file is empty: ${file.absolutePath}" }

        raf.seek(0)

        val magic = ByteArray(6)
        raf.readFully(magic)

        validateMagic(magic, file.name)

        val major = raf.readUnsignedByte()
        val minor = raf.readUnsignedByte()

        val headerLength: Int = when (major) {
            1 -> {
                val b0 = raf.readUnsignedByte()
                val b1 = raf.readUnsignedByte()
                b0 or (b1 shl 8)
            }
            2, 3 -> {
                val b0 = raf.readUnsignedByte()
                val b1 = raf.readUnsignedByte()
                val b2 = raf.readUnsignedByte()
                val b3 = raf.readUnsignedByte()
                b0 or (b1 shl 8) or (b2 shl 16) or (b3 shl 24)
            }
            else -> throw IllegalArgumentException("Unsupported NPY version: $major.$minor")
        }

        val headerBytes = ByteArray(headerLength)
        raf.readFully(headerBytes)
        val header = headerBytes.toString(ASCII)
        val dataOffset = raf.filePointer.toInt()

        return parseHeader(file.name, major, minor, header, dataOffset)
    }

    private fun parseHeader(
        fileName: String,
        major: Int,
        minor: Int,
        header: String,
        dataOffset: Int
    ): Metadata {
        val descr = extractHeaderValue(header, "'descr':")
            ?: throw IllegalArgumentException("NPY header missing descr in $fileName")

        val fortranOrderRaw = extractHeaderValue(header, "'fortran_order':")
            ?: throw IllegalArgumentException("NPY header missing fortran_order in $fileName")

        val shapeRaw = extractShape(header)
            ?: throw IllegalArgumentException("NPY header missing shape in $fileName")

        require(descr.contains("<f4")) {
            "Only little-endian float32 supported. Got descr=$descr in $fileName"
        }

        require(fortranOrderRaw.contains("False")) {
            "Fortran-order arrays not supported in $fileName"
        }

        return Metadata(
            major = major,
            minor = minor,
            shape = parseShape(shapeRaw),
            dataOffset = dataOffset
        )
    }

    private fun validateMagic(bytes: ByteArray, fileName: String) {
        require(bytes[0] == 0x93.toByte()) { "Invalid NPY magic[0] in $fileName" }
        require(bytes[1] == 'N'.code.toByte()) { "Invalid NPY magic[1] in $fileName" }
        require(bytes[2] == 'U'.code.toByte()) { "Invalid NPY magic[2] in $fileName" }
        require(bytes[3] == 'M'.code.toByte()) { "Invalid NPY magic[3] in $fileName" }
        require(bytes[4] == 'P'.code.toByte()) { "Invalid NPY magic[4] in $fileName" }
        require(bytes[5] == 'Y'.code.toByte()) { "Invalid NPY magic[5] in $fileName" }
    }

    private fun extractHeaderValue(header: String, key: String): String? {
        val idx = header.indexOf(key)
        if (idx < 0) return null

        val start = idx + key.length
        val tail = header.substring(start).trimStart()
        return tail.substringBefore(",").trim()
    }

    private fun extractShape(header: String): String? {
        val key = "'shape':"
        val idx = header.indexOf(key)
        if (idx < 0) return null

        val start = idx + key.length
        val tail = header.substring(start)
        val open = tail.indexOf('(')
        val close = tail.indexOf(')')
        if (open < 0 || close < 0 || close <= open) return null

        return tail.substring(open + 1, close)
    }

    private fun parseShape(shapeRaw: String): IntArray {
        return shapeRaw
            .split(",")
            .map { it.trim() }
            .filter { it.isNotEmpty() }
            .map { it.toInt() }
            .toIntArray()
    }
}