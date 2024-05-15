package org.bachelor.classification

import android.Manifest
import android.content.pm.PackageManager
import android.media.AudioRecord
import android.os.Build
import android.os.Bundle
import android.os.Handler
import android.os.HandlerThread
import android.util.Log
import android.view.WindowManager
import androidx.annotation.RequiresApi
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import androidx.core.os.HandlerCompat
import org.tensorflow.lite.task.audio.classifier.ActivityMainBinding
import org.tensorflow.lite.task.audio.classifier.AudioClassifier


class MainActivity : AppCompatActivity() {
  private val probabilitiesAdapter by lazy { ProbabilitiesAdapter() }

  private var audioClassifier: AudioClassifier? = null
  private var audioRecord: AudioRecord? = null
  private var classificationInterval = 500L
  private lateinit var handler: Handler

  override fun onCreate(savedInstanceState: Bundle?) {
    super.onCreate(savedInstanceState)

    val binding = ActivityMainBinding.inflate(layoutInflater)
    setContentView(binding.root)

    with(binding) {
      recyclerView.apply {
        setHasFixedSize(false)
        adapter = probabilitiesAdapter
      }

      keepScreenOn(inputSwitch.isChecked)
      inputSwitch.setOnCheckedChangeListener { _, isChecked ->
        if (isChecked) startAudioClassification() else stopAudioClassification()
        keepScreenOn(isChecked)
      }
    }

    val handlerThread = HandlerThread("backgroundThread")
    handlerThread.start()
    handler = HandlerCompat.createAsync(handlerThread.looper)

    if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
      requestMicrophonePermission()
    } else {
      startAudioClassification()
    }
  }

  private fun startAudioClassification() {
    if (audioClassifier != null) return;

    val classifier = AudioClassifier.createFromFile(this, MODEL_FILE)
    val audioTensor = classifier.createInputTensorAudio()
    val record = classifier.createAudioRecord()
    record.startRecording()

    val run = object : Runnable {
      override fun run() {
        val startTime = System.currentTimeMillis()

        audioTensor.load(record)
        val output = classifier.classify(audioTensor)

//        var flat = output.flatMap { m -> m.categories };
        val filteredDefaultClassesOutput = output[0].categories.filter {
          it.score > 0.8f
        }.sortedBy {
          -it.score
        }

        var hasSilence = filteredDefaultClassesOutput.any { it.label == SILENCE };
        val filteredWeaponClassesOutput = output[1].categories.filter {
          it.score > MINIMUM_DISPLAY_THRESHOLD && !hasSilence
        }.sortedBy {
          -it.score
        }


        val finishTime = System.currentTimeMillis()

        Log.d(TAG, "Latency = ${finishTime - startTime}ms")

        runOnUiThread {
          probabilitiesAdapter.categoryList = filteredWeaponClassesOutput
          probabilitiesAdapter.notifyDataSetChanged()
        }

        handler.postDelayed(this, classificationInterval)
      }
    }

    handler.post(run)

    audioClassifier = classifier
    audioRecord = record
  }

  private fun stopAudioClassification() {
    handler.removeCallbacksAndMessages(null)
    audioRecord?.stop()
    audioRecord = null
    audioClassifier = null
  }

  override fun onTopResumedActivityChanged(isTopResumedActivity: Boolean) {
    if (isTopResumedActivity) {
      startAudioClassification()
    } else {
      stopAudioClassification()
    }
  }

  override fun onRequestPermissionsResult(
          requestCode: Int,
          permissions: Array<out String>,
          grantResults: IntArray
  ) {
    if (requestCode == REQUEST_RECORD_AUDIO) {
      if (grantResults.isNotEmpty() && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
        Log.i(TAG, "Audio permission granted :)")
        startAudioClassification()
      } else {
        Log.e(TAG, "Audio permission not granted :(")
      }
    }
  }

  @RequiresApi(Build.VERSION_CODES.M)
  private fun requestMicrophonePermission() {
    if (ContextCompat.checkSelfPermission(
                    this,
                    Manifest.permission.RECORD_AUDIO
            ) == PackageManager.PERMISSION_GRANTED
    ) {
      startAudioClassification()
    } else {
      requestPermissions(arrayOf(Manifest.permission.RECORD_AUDIO), REQUEST_RECORD_AUDIO)
    }
  }

  private fun keepScreenOn(enable: Boolean) =
    if (enable) {
      window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
    } else {
      window.clearFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
    }

  companion object {
    const val REQUEST_RECORD_AUDIO = 1337
    private const val TAG = "AudioDemo"
    private const val MODEL_FILE = "yamnet.tflite"
    private const val MINIMUM_DISPLAY_THRESHOLD: Float = 0.9f
    private const val SILENCE = "Silence";
  }
}
