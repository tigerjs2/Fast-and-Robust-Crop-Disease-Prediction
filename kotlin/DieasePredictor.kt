package com.example.cropdisease

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Color
import android.graphics.RectF
import kotlin.math.exp
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

data class TensorData(
	val data: FloatArray,
	val shape: LongArray,
)

data class PredictionResult(
	val predictedClass: String,
	val confidence: Float,
	val logits: FloatArray,
	val probabilities: FloatArray,
	val segmentedBitmap: Bitmap,
)

data class DiseasePredictorConfig(
	val samPteAssetPath: String = "models/sam2.1_t_box.pte",
	val featurePteAssetPath: String = "models/prototype_mobv3.pte",
	val textCsvAssetDir: String = "text_embeddings_csv",
	val classNames: List<String> = listOf(
		"cucumber_downy",
		"cucumber_healthy",
		"cucumber_powdery",
		"grape_downy",
		"grape_healthy",
		"paprica_healthy",
		"paprica_powdery",
		"pepper_healthy",
		"pepper_powdery",
		"strawberry_healthy",
		"strawberry_powdery",
		"tomato_graymold",
		"tomato_healthy",
		"tomato_powdery",
	),
	val samInputSize: Int = 1024,
	val featureInputSize: Int = 224,
	val samMaskThreshold: Float = 0.5f,
	val grayBackgroundColor: Int = Color.rgb(128, 128, 128),
	val hardcodedBoxRatio: RectF = RectF(0.15f, 0.15f, 0.85f, 0.85f),
)

interface PteBackend {
	fun run(modelAssetPath: String, inputs: List<TensorData>): List<TensorData>
}

class DiseasePredictor(
	private val context: Context,
	private val backend: PteBackend,
	private val config: DiseasePredictorConfig = DiseasePredictorConfig(),
) {

	private val textEmbeddingsByCrop: Map<String, Array<FloatArray>> =
		loadCropEmbeddingsFromCsv(context, config.textCsvAssetDir)

	private val classPrototypes: Array<FloatArray> by lazy {
		buildClassPrototypes(config.classNames, textEmbeddingsByCrop)
	}

	fun predict(inputBitmap: Bitmap, cropNameRaw: String): PredictionResult {
		val cropName = normalizeCropName(cropNameRaw)
		val cropQueries = textEmbeddingsByCrop[cropName]
			?: error("No CSV embedding found for crop: $cropName")

		val samMask = runSamBoxSegmentation(inputBitmap)
		val segmentedBitmap = applyGrayBackgroundMask(inputBitmap, samMask, config.grayBackgroundColor)
		val imageFeature = runImageFeatureExtractor(segmentedBitmap)

		val fusedFeature = CrossAttentionOps.singleQueryCrossAttention(
			imageFeature = imageFeature,
			textQueries = cropQueries,
		)

		val logits = classifyWithPrototypes(fusedFeature, classPrototypes)
		val probs = softmax(logits)
		val topIdx = argmax(probs)

		return PredictionResult(
			predictedClass = config.classNames[topIdx],
			confidence = probs[topIdx],
			logits = logits,
			probabilities = probs,
			segmentedBitmap = segmentedBitmap,
		)
	}

	private fun runSamBoxSegmentation(bitmap: Bitmap): BooleanArray {
		val resized = Bitmap.createScaledBitmap(bitmap, config.samInputSize, config.samInputSize, true)
		val imageTensor = bitmapToChwFloatTensor(resized, normalize = false)

		val ratio = config.hardcodedBoxRatio
		val x1 = ratio.left * config.samInputSize
		val y1 = ratio.top * config.samInputSize
		val x2 = ratio.right * config.samInputSize
		val y2 = ratio.bottom * config.samInputSize
		val boxTensor = floatArrayOf(x1, y1, x2, y2)

		val outputs = backend.run(
			modelAssetPath = config.samPteAssetPath,
			inputs = listOf(
				TensorData(
					data = imageTensor,
					shape = longArrayOf(1, 3, config.samInputSize.toLong(), config.samInputSize.toLong()),
				),
				TensorData(
					data = boxTensor,
					shape = longArrayOf(1, 4),
				),
			),
		)

		val maskTensor = outputs.firstOrNull() ?: error("SAM output is empty")
		return decodeMaskToOriginalSize(
			maskTensor = maskTensor,
			srcWidth = config.samInputSize,
			srcHeight = config.samInputSize,
			dstWidth = bitmap.width,
			dstHeight = bitmap.height,
			threshold = config.samMaskThreshold,
		)
	}

	private fun runImageFeatureExtractor(segmentedBitmap: Bitmap): FloatArray {
		val resized = Bitmap.createScaledBitmap(
			segmentedBitmap,
			config.featureInputSize,
			config.featureInputSize,
			true,
		)
		val imageTensor = bitmapToChwFloatTensor(resized, normalize = true)

		val outputs = backend.run(
			modelAssetPath = config.featurePteAssetPath,
			inputs = listOf(
				TensorData(
					data = imageTensor,
					shape = longArrayOf(1, 3, config.featureInputSize.toLong(), config.featureInputSize.toLong()),
				),
			),
		)

		val featureTensor = outputs.firstOrNull() ?: error("Feature extractor output is empty")
		return flattenFeature(featureTensor)
	}

	private fun classifyWithPrototypes(feature: FloatArray, prototypes: Array<FloatArray>): FloatArray {
		val f = l2Normalize(feature)
		return FloatArray(prototypes.size) { idx ->
			dot(f, prototypes[idx])
		}
	}
}

object CrossAttentionOps {
	fun singleQueryCrossAttention(
		imageFeature: FloatArray,
		textQueries: Array<FloatArray>,
	): FloatArray {
		val q = l2Normalize(imageFeature)
		val keys = textQueries.map { l2Normalize(it) }
		val values = textQueries.map { it }

		val scale = sqrt(max(1, q.size).toFloat())
		val scores = FloatArray(keys.size) { i -> dot(q, keys[i]) / scale }
		val attn = softmax(scores)

		val context = FloatArray(imageFeature.size)
		for (i in values.indices) {
			val w = attn[i]
			val v = values[i]
			for (d in context.indices) {
				context[d] += w * v[d]
			}
		}

		val fused = FloatArray(imageFeature.size)
		for (d in fused.indices) {
			fused[d] = imageFeature[d] + context[d]
		}
		return l2Normalize(fused)
	}
}

fun loadCropEmbeddingsFromCsv(context: Context, assetDir: String): Map<String, Array<FloatArray>> {
	val fileNames = context.assets.list(assetDir)?.toList().orEmpty()
	val csvFiles = fileNames.filter { it.endsWith(".csv", ignoreCase = true) }

	val map = linkedMapOf<String, Array<FloatArray>>()
	for (fileName in csvFiles) {
		val crop = fileName.substringBeforeLast(".").lowercase()
		val rows = mutableListOf<FloatArray>()

		context.assets.open("$assetDir/$fileName").bufferedReader().use { br ->
			val lines = br.readLines()
			if (lines.size <= 1) {
				return@use
			}
			for (line in lines.drop(1)) {
				if (line.isBlank()) continue
				val cols = line.split(',')
				if (cols.size <= 2) continue
				val vec = FloatArray(cols.size - 1)
				for (i in 1 until cols.size) {
					vec[i - 1] = cols[i].toFloat()
				}
				rows += vec
			}
		}

		if (rows.isNotEmpty()) {
			map[crop] = rows.toTypedArray()
		}
	}

	if (map.isEmpty()) {
		error("No embedding CSV found in assets/$assetDir")
	}
	return map
}

fun buildClassPrototypes(
	classNames: List<String>,
	textEmbeddingsByCrop: Map<String, Array<FloatArray>>,
): Array<FloatArray> {
	val dim = textEmbeddingsByCrop.values.first().first().size
	return Array(classNames.size) { classIdx ->
		val className = classNames[classIdx]
		val crop = normalizeCropName(className.substringBefore('_'))
		val queries = textEmbeddingsByCrop[crop]
			?: error("No crop embedding found for class prototype: $className")

		val mean = FloatArray(dim)
		for (q in queries) {
			for (d in 0 until dim) {
				mean[d] += q[d]
			}
		}
		for (d in 0 until dim) {
			mean[d] /= queries.size.toFloat()
		}
		l2Normalize(mean)
	}
}

fun applyGrayBackgroundMask(bitmap: Bitmap, mask: BooleanArray, grayColor: Int): Bitmap {
	val w = bitmap.width
	val h = bitmap.height
	require(mask.size == w * h) { "Mask size mismatch. expected=${w * h}, got=${mask.size}" }

	val out = bitmap.copy(Bitmap.Config.ARGB_8888, true)
	val pixels = IntArray(w * h)
	out.getPixels(pixels, 0, w, 0, 0, w, h)

	for (i in pixels.indices) {
		if (!mask[i]) {
			pixels[i] = grayColor
		}
	}
	out.setPixels(pixels, 0, w, 0, 0, w, h)
	return out
}

fun decodeMaskToOriginalSize(
	maskTensor: TensorData,
	srcWidth: Int,
	srcHeight: Int,
	dstWidth: Int,
	dstHeight: Int,
	threshold: Float,
): BooleanArray {
	val srcMask = extractMask2d(maskTensor, srcWidth, srcHeight)
	val dstMask = BooleanArray(dstWidth * dstHeight)

	for (y in 0 until dstHeight) {
		val sy = min(srcHeight - 1, (y.toFloat() / dstHeight * srcHeight).toInt())
		for (x in 0 until dstWidth) {
			val sx = min(srcWidth - 1, (x.toFloat() / dstWidth * srcWidth).toInt())
			val v = srcMask[sy * srcWidth + sx]
			dstMask[y * dstWidth + x] = v >= threshold
		}
	}
	return dstMask
}

fun extractMask2d(maskTensor: TensorData, width: Int, height: Int): FloatArray {
	val total = width * height
	val shape = maskTensor.shape
	val data = maskTensor.data

	if (data.size == total) {
		return data
	}

	if (shape.size == 4L.toInt() && shape[0] == 1L && shape[1] == 1L &&
		shape[2].toInt() == height && shape[3].toInt() == width
	) {
		return data.copyOfRange(0, total)
	}

	if (data.size >= total) {
		return data.copyOfRange(0, total)
	}
	error("Unsupported SAM mask shape=${shape.contentToString()} dataSize=${data.size}")
}

fun flattenFeature(tensor: TensorData): FloatArray {
	val data = tensor.data
	val shape = tensor.shape

	if (shape.isEmpty()) return data
	if (shape.size == 2 && shape[0] == 1L) return data.copyOfRange(0, shape[1].toInt())
	if (shape.size == 4 && shape[0] == 1L) {
		val c = shape[1].toInt()
		val h = shape[2].toInt()
		val w = shape[3].toInt()
		val out = FloatArray(c)
		val spatial = max(1, h * w)
		for (ci in 0 until c) {
			var sum = 0f
			for (i in 0 until spatial) {
				sum += data[ci * spatial + i]
			}
			out[ci] = sum / spatial.toFloat()
		}
		return out
	}
	return data
}

fun bitmapToChwFloatTensor(bitmap: Bitmap, normalize: Boolean): FloatArray {
	val w = bitmap.width
	val h = bitmap.height
	val pixels = IntArray(w * h)
	bitmap.getPixels(pixels, 0, w, 0, 0, w, h)

	val out = FloatArray(3 * w * h)
	val area = w * h

	val mean = floatArrayOf(0.485f, 0.456f, 0.406f)
	val std = floatArrayOf(0.229f, 0.224f, 0.225f)

	for (i in pixels.indices) {
		val p = pixels[i]
		var r = Color.red(p) / 255.0f
		var g = Color.green(p) / 255.0f
		var b = Color.blue(p) / 255.0f

		if (normalize) {
			r = (r - mean[0]) / std[0]
			g = (g - mean[1]) / std[1]
			b = (b - mean[2]) / std[2]
		}

		out[i] = r
		out[area + i] = g
		out[2 * area + i] = b
	}
	return out
}

fun normalizeCropName(raw: String): String {
	val n = raw.trim().lowercase()
	return if (n == "paprica") "paprika" else n
}

fun l2Normalize(v: FloatArray): FloatArray {
	var normSq = 0f
	for (x in v) normSq += x * x
	val norm = sqrt(max(normSq, 1e-12f))
	return FloatArray(v.size) { i -> v[i] / norm }
}

fun dot(a: FloatArray, b: FloatArray): Float {
	require(a.size == b.size) { "Dot size mismatch: ${a.size} vs ${b.size}" }
	var s = 0f
	for (i in a.indices) s += a[i] * b[i]
	return s
}

fun softmax(x: FloatArray): FloatArray {
	if (x.isEmpty()) return x
	var maxVal = x[0]
	for (v in x) if (v > maxVal) maxVal = v

	val exps = FloatArray(x.size)
	var sum = 0f
	for (i in x.indices) {
		val e = exp((x[i] - maxVal).toDouble()).toFloat()
		exps[i] = e
		sum += e
	}
	if (sum <= 0f) return FloatArray(x.size) { 1f / x.size }
	return FloatArray(x.size) { i -> exps[i] / sum }
}

fun argmax(x: FloatArray): Int {
	require(x.isNotEmpty()) { "Cannot argmax empty array" }
	var best = 0
	for (i in 1 until x.size) {
		if (x[i] > x[best]) best = i
	}
	return best
}
