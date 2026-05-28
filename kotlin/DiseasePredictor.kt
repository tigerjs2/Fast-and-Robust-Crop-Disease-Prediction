class DiseasePredictor(private val context: Context) {
    companion object {
        private const val BACKEND_URL = "https://trio-handed-promptly.ngrok-free.dev"
    }

    private var classNames: List<String> = emptyList()
    private val retrofit = Retrofit.Builder()
        .baseUrl(BACKEND_URL)
        .addConverterFactory(GsonConverterFactory.create())
        .build()
    
    private val apiService = retrofit.create(DiseaseApi::class.java)

    fun initModels(classNamesAsset: String = "class_names.json") {
        // 로컬에서는 class_names만 로드 (UI dropdown용)
        classNames = loadJsonArray(classNamesAsset)
    }

    suspend fun predictWithImage(
        bitmap: Bitmap, 
        cropName: String
    ): PredictionResult {
        val multipartBody = MultipartBody.Builder()
            .setType(MultipartBody.FORM)
            .addFormDataPart("crop_name", cropName)
            .addFormDataPart(
                "image", 
                "photo.jpg",
                RequestBody.create("image/jpeg".toMediaType(), bitmapToByteArray(bitmap))
            )
            .build()

        val response = apiService.predict(multipartBody)
        return PredictionResult(
            className = response.predicted_class,
            confidence = response.confidence.toFloat()
        )
    }

    private fun bitmapToByteArray(bitmap: Bitmap): ByteArray {
        val stream = ByteArrayOutputStream()
        bitmap.compress(Bitmap.CompressFormat.JPEG, 90, stream)
        return stream.toByteArray()
    }

    private fun loadJsonArray(assetName: String): List<String> {
        val text = readAssetText(assetName)
        val array = JSONArray(text)
        return (0 until array.length()).map { array.getString(it) }
    }

    private fun readAssetText(assetName: String): String {
        return context.assets.open(assetName).bufferedReader().use { it.readText() }
    }
}

// Retrofit API 인터페이스
interface DiseaseApi {
    @Multipart
    @POST("/predict")
    suspend fun predict(
        @Part body: MultipartBody
    ): PredictionResponse
}

data class PredictionResponse(
    val predicted_class: String,
    val confidence: Double
)
