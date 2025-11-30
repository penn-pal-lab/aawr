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

// Global handler for abstract toggle (used by onclick in HTML)
function toggleAbstract() {
  var content = document.getElementById('abstract-content');
  var triangle = document.getElementById('abstract-triangle');
  var hint = document.getElementById('abstract-hint');
  if (!content || !triangle) return;

  if (content.style.display === 'none') {
    content.style.display = 'block';
    // Rotate triangle to point up (▲)
    triangle.style.borderTop = 'none';
    triangle.style.borderBottom = '8px solid #3273dc';
    // Update hint text
    if (hint) hint.textContent = '(click to collapse)';
  } else {
    content.style.display = 'none';
    // Rotate triangle to point down (▼)
    triangle.style.borderTop = '8px solid #3273dc';
    triangle.style.borderBottom = 'none';
    // Update hint text
    if (hint) hint.textContent = '(click to expand)';
  }
}

// 3D trajectory viewer controls
function initTrajectoryControls() {
  var trajRoot = document.getElementById('traj-controls');
  if (!trajRoot) {
    return;
  }

  var state = {
    mode: 'offline',
    scene: 'bookshelf',
    object: 'pineapple',
    algo: 'aawr'
  };

  var groupScene = document.getElementById('traj-group-scene');
  var groupObject = document.getElementById('traj-group-object');
  var groupAlgo = document.getElementById('traj-group-algo');

  function setSelectedInGroup(group, value) {
    var buttons = document.querySelectorAll('.traj-option[data-group="' + group + '"]');
    buttons.forEach(function(btn) {
      var isSelected = btn.dataset.value === value;
      btn.classList.toggle('is-selected', isSelected);
      if (isSelected) {
        btn.classList.remove('is-light');
        btn.classList.add('is-link');
      } else {
        btn.classList.remove('is-link');
        btn.classList.add('is-light');
      }
    });
  }

  function setButtonDisabled(btn, disabled) {
    if (disabled) {
      btn.classList.add('is-disabled');
      btn.disabled = true;
    } else {
      btn.classList.remove('is-disabled');
      btn.disabled = false;
    }
  }

  function getTrajectoryBase(selection) {
    if (selection.mode === 'offline') {
      if (selection.scene === 'bookshelf') {
        if (selection.object === 'pineapple') return 'offline_bookshelf_p';
        if (selection.object === 'duck') return 'offline_bookshelf_d';
      }
      if (selection.scene === 'shelf_cabinet' && selection.object === 'pineapple') {
        return 'offline_shelf_cabinet';
      }
      if (selection.scene === 'complex' && selection.object === 'pineapple') {
        return 'offline_complex';
      }
      return null;
    }

    // Online mode: AAWR / AWR / BC use global rollouts;
    // Exhaustive uses scene-specific exhaustive trajectories.
    if (selection.algo === 'aawr') return 'online_aawr';
    if (selection.algo === 'awr') return 'online_awr';
    if (selection.algo === 'bc') return 'online_bc';
    if (selection.algo === 'exhaustive') {
      if (selection.scene === 'bookshelf') return 'exhaustive_bookshelf';
      if (selection.scene === 'shelf_cabinet') return 'exhaustive_shelf_cabinet';
      if (selection.scene === 'complex') return 'exhaustive_complex';
    }
    return null;
  }

  function updateAvailability(selection) {
    var mode = selection.mode;

    // Algo group only matters in online mode
    if (groupAlgo) {
      if (mode === 'online') {
        groupAlgo.classList.remove('is-hidden');
      } else {
        groupAlgo.classList.add('is-hidden');
      }
    }

    // Scene group is always shown; for online non-exhaustive algos it's visually
    // allowed but does not change files.

    // Object group: used only in offline mode.
    if (groupObject) {
      if (mode === 'offline') {
        groupObject.classList.remove('is-hidden');
        var objectButtons = document.querySelectorAll('.traj-option[data-group="object"]');
        var hasValidObject = false;
        objectButtons.forEach(function(btn) {
          var value = btn.dataset.value;
          var disabled = false;
          if ((selection.scene === 'shelf_cabinet' || selection.scene === 'complex') && value === 'duck') {
            disabled = true;
          }
          setButtonDisabled(btn, disabled);
          if (!disabled && value === selection.object) {
            hasValidObject = true;
          }
        });

        if (!hasValidObject) {
          var objectButtonsArr = Array.prototype.slice.call(
            document.querySelectorAll('.traj-option[data-group="object"]')
          );
          var fallback = objectButtonsArr.find(function(btn) {
            return !btn.classList.contains('is-disabled');
          });
          if (fallback) {
            selection.object = fallback.dataset.value;
            setSelectedInGroup('object', selection.object);
          }
        }
      } else {
        groupObject.classList.add('is-hidden');
      }
    }
  }

  function updateMedia(selection) {
    var base = getTrajectoryBase(selection);
    var img = document.getElementById('traj-image');
    var video = document.getElementById('traj-video');
    if (!img || !video || !base) {
      return;
    }

    var pngPath = 'static/videos/3dtraj/' + base + '.png';
    var mp4Path = 'static/videos/3dtraj/' + base + '.mp4';

    img.src = pngPath;
    video.src = mp4Path;
    video.load();
    if (video.paused) {
      video.play().catch(function() { });
    }
  }

  function refreshAll() {
    updateAvailability(state);
    updateMedia(state);
  }

  var buttons = document.querySelectorAll('.traj-option');
  buttons.forEach(function(btn) {
    btn.addEventListener('click', function() {
      if (btn.classList.contains('is-disabled')) {
        return;
      }
      var group = btn.dataset.group;
      var value = btn.dataset.value;
      if (!group || !value) {
        return;
      }
      state[group] = value;
      setSelectedInGroup(group, value);
      refreshAll();
    });
  });

  // Initial render
  setSelectedInGroup('mode', state.mode);
  setSelectedInGroup('scene', state.scene);
  setSelectedInGroup('object', state.object);
  setSelectedInGroup('algo', state.algo);
  refreshAll();
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
    initTrajectoryControls();

})
