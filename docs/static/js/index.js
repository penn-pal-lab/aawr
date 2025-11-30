window.HELP_IMPROVE_VIDEOJS = false;

var INTERP_BASE = "./static/interpolation/stacked";
var NUM_INTERP_FRAMES = 240;

var interp_images = [];
function preloadInterpolationImages() {
  for (var i = 0; i < NUM_INTERP_FRAMES; i++) {
    var path = INTERP_BASE + '/' + String(i).padStart(6, '0') + '.jpg';
    interp_images[i] = new Image();
    interp_images[i].src = path;
  }
}

function setInterpolationImage(i) {
  var image = interp_images[i];
  image.ondragstart = function() { return false; };
  image.oncontextmenu = function() { return false; };
  $('#interpolation-image-wrapper').empty().append(image);
}

// VGGT wrist camera scan viewer configuration
var VGGT_VIDEOS = {
  longhorizon: {
    aawr: 'static/videos/vggt/pair1-longhorizon_AAWR_success_multiview.mp4',
    awr: 'static/videos/vggt/pair1-longhorizon_AWR_failure_multiview.mp4',
    bc: 'static/videos/vggt/pair1-longhorizon_BC_failure_multiview.mp4',
    pi0: 'static/videos/vggt/pair1-longhorizon_pi0_failure_multiview.mp4',
    teleoperation: 'static/videos/vggt/pair1-longhorizon_teleoperation_success_multiview.mp4'
  },
  fixation: {
    aawr: 'static/videos/vggt/pair2-fixation_AAWR_success_multiview.mp4',
    awr: 'static/videos/vggt/pair2-fixation_AWR_failure_multiview.mp4',
    bc: 'static/videos/vggt/pair2-fixation_BC_failure_multiview.mp4',
    pi0: 'static/videos/vggt/pair2-fixation_pi0_failure_multiview.mp4'
    // no teleoperation fixation video
  }
};

function initVggtViewer() {
  var videoEl = document.getElementById('vggt-video');
  if (!videoEl) return;

  var behaviorButtons = document.querySelectorAll('[data-vggt-behavior]');
  var algoButtons = document.querySelectorAll('[data-vggt-algo]');
  var statusEl = document.getElementById('vggt-status');

  var currentBehavior = 'longhorizon';
  var currentAlgo = 'aawr';

  function hasVideo(behavior, algo) {
    return VGGT_VIDEOS[behavior] && VGGT_VIDEOS[behavior][algo];
  }

  function updateAlgoButtons() {
    algoButtons.forEach(function(btn) {
      var algo = btn.getAttribute('data-vggt-algo');
      var available = hasVideo(currentBehavior, algo);
      btn.disabled = !available;
      btn.classList.toggle('is-static', !available);
      btn.classList.toggle('is-light', !available || algo !== currentAlgo);
      btn.classList.toggle('is-info', available && algo === currentAlgo);
    });
  }

  function setVideoSource() {
    if (!hasVideo(currentBehavior, currentAlgo)) return;
    var src = VGGT_VIDEOS[currentBehavior][currentAlgo];
    if (src && videoEl.getAttribute('src') !== src) {
      videoEl.setAttribute('src', src);
      videoEl.load();
      var playPromise = videoEl.play();
      if (playPromise && typeof playPromise.then === 'function') {
        playPromise.catch(function() {
          // Autoplay might be blocked; ignore.
        });
      }
    }
    if (statusEl) {
      var behaviorBtn = document.querySelector('[data-vggt-behavior="' + currentBehavior + '"]');
      var algoBtn = document.querySelector('[data-vggt-algo="' + currentAlgo + '"]');
      var behaviorLabel = behaviorBtn ? behaviorBtn.textContent.trim() : currentBehavior;
      var algoLabel = algoBtn ? algoBtn.textContent.trim() : currentAlgo;
      statusEl.textContent = behaviorLabel + ' • ' + algoLabel;
    }
  }

  behaviorButtons.forEach(function(btn) {
    btn.addEventListener('click', function() {
      var behavior = this.getAttribute('data-vggt-behavior');
      if (!VGGT_VIDEOS[behavior]) return;
      currentBehavior = behavior;
      behaviorButtons.forEach(function(b) {
        var isActive = b.getAttribute('data-vggt-behavior') === currentBehavior;
        b.classList.toggle('is-info', isActive);
        b.classList.toggle('is-light', !isActive);
      });

      if (!hasVideo(currentBehavior, currentAlgo)) {
        // Fallback to the first available algorithm for this behavior
        var algos = Object.keys(VGGT_VIDEOS[currentBehavior]);
        for (var i = 0; i < algos.length; i++) {
          if (hasVideo(currentBehavior, algos[i])) {
            currentAlgo = algos[i];
            break;
          }
        }
      }
      updateAlgoButtons();
      setVideoSource();
    });
  });

  algoButtons.forEach(function(btn) {
    btn.addEventListener('click', function() {
      if (this.disabled) return;
      currentAlgo = this.getAttribute('data-vggt-algo');
      updateAlgoButtons();
      setVideoSource();
    });
  });

  // Initialize state
  updateAlgoButtons();
  setVideoSource();
}

// Apply per-video playback rates based on a data attribute in HTML:
// <video ... data-playback-rate="0.066"></video>
function initCustomPlaybackRates() {
  var videos = document.querySelectorAll('video[data-playback-rate]');

  videos.forEach(function(video) {
    var rate = parseFloat(video.getAttribute('data-playback-rate'));
    if (!rate || rate <= 0) return;

    function applyRate() {
      video.playbackRate = rate;
    }

    if (video.readyState >= 1) {
      applyRate();
    } else {
      video.addEventListener('loadedmetadata', applyRate);
    }
  });
}


$(document).ready(function() {
    // Check for click events on the navbar burger icon
    $(".navbar-burger").click(function() {
      // Toggle the "is-active" class on both the "navbar-burger" and the "navbar-menu"
      $(".navbar-burger").toggleClass("is-active");
      $(".navbar-menu").toggleClass("is-active");

    });

    var options = {
			slidesToScroll: 1,
			slidesToShow: 5,
			loop: true,
			infinite: true,
			autoplay: true,
			autoplaySpeed: 3000,
    }

		// Initialize all div with carousel class
    var carousels = bulmaCarousel.attach('.envs-carousel', options);

    // Loop on each carousel initialized
    for(var i = 0; i < carousels.length; i++) {
    	// Add listener to  event
    	carousels[i].on('before:show', state => {
    		console.log(state);
    	});
    }

    // Access to bulmaCarousel instance of an element
    var element = document.querySelector('#my-element');
    if (element && element.bulmaCarousel) {
    	// bulmaCarousel instance is available as element.bulmaCarousel
    	element.bulmaCarousel.on('before-show', function(state) {
    		console.log(state);
    	});
    }

    /*var player = document.getElementById('interpolation-video');
    player.addEventListener('loadedmetadata', function() {
      $('#interpolation-slider').on('input', function(event) {
        console.log(this.value, player.duration);
        player.currentTime = player.duration / 100 * this.value;
      })
    }, false);*/
    // preloadInterpolationImages();

    // $('#interpolation-slider').on('input', function(event) {
    //   setInterpolationImage(this.value);
    // });
    // setInterpolationImage(0);
    // $('#interpolation-slider').prop('max', NUM_INTERP_FRAMES - 1);

    bulmaSlider.attach();
    initVggtViewer();
    initCustomPlaybackRates();

})
