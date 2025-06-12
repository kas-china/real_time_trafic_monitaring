$(document).ready(function() {
    // Initialize toast
    const alertToast = new bootstrap.Toast(document.getElementById('alert-toast'));

    // Global state
    let isCameraRunning = false;
    let currentCameraIndex = 0;
    let lastAlertId = 0;

    // Show toast notification
    function showToast(message, type = 'info') {
        const toast = $('#alert-toast');
        const title = $('#toast-title');
        const messageEl = $('#alert-message');
        toast.removeClass('bg-success bg-danger bg-warning bg-info');
        title.text(type.charAt(0).toUpperCase() + type.slice(1));
        messageEl.text(message);
        toast.addClass(`bg-${type}`);
        alertToast.show();
    }

    // Validate file extension
    function allowedFile(filename) {
        const allowedExtensions = ['png', 'jpg', 'jpeg', 'mp4', 'avi', 'mov'];
        const ext = filename.split('.').pop().toLowerCase();
        return allowedExtensions.includes(ext);
    }

    // Update stats
    function updateStats() {
        $('.stat-card h3').each(function() {
            $(this).html('<span class="spinner-border spinner-border-sm"></span>');
        });

        $.get('/get_counts')
            .done(function(data) {
                if (data && data.counts && data.status === 'success') {
                    $('#total-count').fadeOut(200, function() {
                        $(this).text(data.total || 0).fadeIn(200);
                    });
                    $('#car-count').fadeOut(200, function() {
                        $(this).text(data.counts.car || 0).fadeIn(200);
                    });
                    $('#truck-count').fadeOut(200, function() {
                        $(this).text(data.counts.truck || 0).fadeIn(200);
                    });
                    $('#bus-count').fadeOut(200, function() {
                        $(this).text(data.counts.bus || 0).fadeIn(200);
                    });
                    $('#motorcycle-count').fadeOut(200, function() {
                        $(this).text(data.counts.motorcycle || 0).fadeIn(200);
                    });
                    $('#stats-last-updated').text('Updated: ' + new Date().toLocaleTimeString());
                } else {
                    showToast('Data format error', 'warning');
                }
            })
            .fail(function() {
                $('.stat-card h3').each(function() {
                    $(this).html('<i class="bi bi-exclamation-triangle"></i>');
                });
                showToast('Failed to update stats', 'danger');
            });
    }

    // Load detection history
    function loadHistory() {
        $('#history-data').html(`
            <tr>
                <td colspan="9" class="text-center py-4">
                    <div class="spinner-border text-primary" role="status">
                        <span class="visually-hidden">Loading...</span>
                    </div>
                    <p class="mt-2">Loading history...</p>
                </td>
            </tr>
        `);

        $.get('/get_history')
            .done(function(data) {
                let html = '';
                if (data.length > 0) {
                    data.forEach(item => {
                        html += `
                        <tr>
                            <td>${item.id}</td>
                            <td>${item.date}</td>
                            <td><span class="badge ${item.type === 'video' ? 'bg-primary' : 'bg-info'}">${item.type}</span></td>
                            <td>${item.car || 0}</td>
                            <td>${item.truck || 0}</td>
                            <td>${item.bus || 0}</td>
                            <td>${item.motorcycle || 0}</td>
                            <td>${item.total || 0}</td>
                            <td>
                                <button class="btn btn-sm btn-outline-primary view-detection" data-id="${item.id}">
                                    <i class="bi bi-eye"></i> View
                                </button>
                            </td>
                        </tr>`;
                    });
                } else {
                    html = `<tr><td colspan="9" class="text-center py-4 text-muted">No detection history available</td></tr>`;
                }
                $('#history-data').html(html);
            })
            .fail(function() {
                $('#history-data').html(`
                    <tr>
                        <td colspan="9" class="text-center py-4 text-danger">
                        Failed to load detection history
                    </td>
                </tr>
            `);
            showToast('Failed to load detection history', 'danger');
        });
    }

    // Load alerts
    function loadAlerts() {
        $.get('/get_alerts')
            .done(function(data) {
                let html = '';
                if (data.length > 0) {
                    data.forEach(alert => {
                        let rowClass = alert.alert_type === 'threshold' ? 'threshold-alert' : 'unallowed-alert';
                        if (alert.id > lastAlertId) {
                            rowClass += ' new-alert';
                            setTimeout(() => {
                                $(`#alert-${alert.id}`).removeClass('new-alert');
                            }, 5000);
                        }
                        html += `<tr id="alert-${alert.id}" class="alert-row ${rowClass}">`;
                        html += `<td>${alert.id}</td>`;
                        html += `<td>${alert.timestamp}</td>`;
                        html += `<td>${alert.alert_type}</td>`;
                        html += `<td>${alert.message}</td>`;
                        html += `</tr>`;
                        if (alert.id > lastAlertId) {
                            lastAlertId = alert.id;
                        }
                    });
                } else {
                    html = `<tr><td colspan="4" class="text-center py-4 text-muted">No alerts available</td></tr>`;
                }
                $('#alert-table').html(html);
            })
            .fail(function() {
                $('#alert-table').html(`
                    <tr>
                        <td colspan="4" class="text-center py-4 text-danger">
                        Failed to load alerts
                    </td>
                </tr>
            `);
            showToast('Failed to load alerts', 'danger');
        });
    }

    // View detection details
    $(document).on('click', '.view-detection', function() {
        const detectionId = $(this).data('id');
        $.get(`/get_detection_details/${detectionId}`)
            .done(function(response) {
                if (response.status === 'success') {
                    const data = response.data;
                    $('#modal-id').text(data.id);
                    $('#modal-date').text(data.date);
                    $('#modal-type').text(data.type);
                    $('#modal-car').text(data.car || 0);
                    $('#modal-truck').text(data.truck || 0);
                    $('#modal-bus').text(data.bus || 0);
                    $('#modal-motorcycle').text(data.motorcycle || 0);
                    $('#modal-total').text(data.total || 0);
                    if (data.file_path) {
                        if (data.type === 'image') {
                            $('#modal-media').html(`<img src="${data.file_path}" alt="Processed Image" class="img-fluid">`);
                        } else if (data.type === 'video') {
                            $('#modal-media').html(`
                                <video controls class="img-fluid">
                                    <source src="${data.file_path}" type="video/mp4">
                                    Your browser does not support the video tag.
                                </video>
                            `);
                        }
                    } else {
                        $('#modal-media').html('<p class="text-muted">No media available</p>');
                    }
                    $('#detectionModal').modal('show');
                } else {
                    showToast(response.message || 'Failed to load detection details', 'danger');
                }
            })
            .fail(function() {
                showToast('Failed to load detection details', 'danger');
            });
    });

    // Export functions
    $('#export-csv').click(function() {
        window.location.href = '/export_csv';
    });

    $('#export-pdf').click(function() {
        window.location.href = '/export_pdf';
    });

    // Chart loading functions
    function loadRadarChart() {
        $('#radar-chart').attr('src', '');
        $('#radar-chart-container').html('<div class="text-center py-4"><div class="spinner-border text-primary" role="status"></div><p>Loading radar chart...</p></div>');
        
        $.get('/radar_chart')
            .done(function(data) {
                if (data.status === 'success') {
                    $('#radar-chart-container').html(`<img src="data:image/png;base64,${data.image}" id="radar-chart" class="chart-img" alt="Radar Chart">`);
                } else {
                    $('#radar-chart-container').html('<div class="alert alert-danger">Failed to load radar chart</div>');
                }
            })
            .fail(function() {
                $('#radar-chart-container').html('<div class="alert alert-danger">Failed to load radar chart</div>');
            });
    }

    function loadLineChart() {
        $('#line-chart').attr('src', '');
        $('#line-chart-container').html('<div class="text-center py-4"><div class="spinner-border text-primary" role="status"></div><p>Loading line chart...</p></div>');
        
        $.get('/line_chart')
            .done(function(data) {
                if (data.status === 'success') {
                    $('#line-chart-container').html(`<img src="data:image/png;base64,${data.image}" id="line-chart" class="chart-img" alt="Line Chart">`);
                } else {
                    $('#line-chart-container').html('<div class="alert alert-danger">Failed to load line chart</div>');
                }
            })
            .fail(function() {
                $('#line-chart-container').html('<div class="alert alert-danger">Failed to load line chart</div>');
            });
    }

    function loadBarChart() {
        $('#bar-chart').attr('src', '');
        $('#bar-chart-container').html('<div class="text-center py-4"><div class="spinner-border text-primary" role="status"></div><p>Loading bar chart...</p></div>');
        
        $.get('/bar_chart')
            .done(function(data) {
                if (data.status === 'success') {
                    $('#bar-chart-container').html(`<img src="data:image/png;base64,${data.image}" id="bar-chart" class="chart-img" alt="Bar Chart">`);
                } else {
                    $('#bar-chart-container').html('<div class="alert alert-danger">Failed to load bar chart</div>');
                }
            })
            .fail(function() {
                $('#bar-chart-container').html('<div class="alert alert-danger">Failed to load bar chart</div>');
            });
    }

    // Chart download functions
    $('#download-radar').click(function() {
        window.location.href = '/download_chart/radar';
    });

    $('#download-line').click(function() {
        window.location.href = '/download_chart/line';
    });

    $('#download-bar').click(function() {
        window.location.href = '/download_chart/bar';
    });

    // Real-time detection log updates
    function updateDetectionLog() {
        $.get('/get_recent_detections')
            .done(function(detections) {
                const logContainer = $('#detection-log');
                if (detections.length === 0) return;

                if ($('#detection-log .text-muted').length) {
                    logContainer.html('');
                }

                detections.forEach(detection => {
                    const time = new Date(detection.timestamp).toLocaleTimeString();
                    const counts = [
                        { type: 'car', count: detection.car || 0, icon: 'bi-car-front', color: 'success' },
                        { type: 'truck', count: detection.truck || 0, icon: 'bi-truck', color: 'warning' },
                        { type: 'bus', count: detection.bus || 0, icon: 'bi-bus-front', color: 'info' },
                        { type: 'motorcycle', count: detection.motorcycle || 0, icon: 'bi-bicycle', color: 'danger' }
                    ].filter(item => item.count > 0);

                    if (counts.length > 0) {
                        counts.forEach(item => {
                            logContainer.prepend(`
                                <div class="list-group-item detection-log-item">
                                    <div class="d-flex justify-content-between">
                                        <span><i class="bi ${item.icon} text-${item.color} me-2"></i>
                                            ${item.count} ${item.type}${item.count > 1 ? 's' : ''} detected</span>
                                        <span class="badge bg-${detection.type === 'video' ? 'primary' : 'info'}">${detection.type}</span>
                                    </div>
                                    <small class="text-muted">${time}</small>
                                </div>
                            `);
                        });
                    }

                    // Limit to 50 items
                    const $items = $('#detection-log .list-group-item');
                    if ($items.length > 50) {
                        $items.slice(50).remove();
                    }
                });
                $('#last-updated').text('Updated: ' + new Date().toLocaleTimeString());
            })
            .fail(function() {
                showToast('Failed to update detection log', 'danger');
            });
    }

    // Camera controls
    function startCamera() {
        if (isCameraRunning) return;

        $('#start-camera').prop('disabled', true);
        $('#stop-camera').prop('disabled', false);

        $.get('/start_camera')
            .done(function(response) {
                if (response.status === 'success') {
                    isCameraRunning = true;
                    $('#video-feed').attr('src', `/video_feed?t=${new Date().getTime()}`);
                    showToast('Camera started successfully', 'success');
                } else {
                    showToast(response.message || 'Failed to start camera', 'danger');
                    $('#start-camera').prop('disabled', false);
                }
            })
            .fail(function() {
                showToast('Failed to start camera', 'danger');
                $('#start-camera').prop('disabled', false);
            });
    }

    function stopCamera() {
        if (!isCameraRunning) return;

        $('#stop-camera').prop('disabled', true);

        $.get('/stop_camera')
            .done(function(response) {
                if (response.status === 'success') {
                    isCameraRunning = false;
                    $('#video-feed').attr('src', '');
                    setTimeout(() => $('#video-feed').attr('src', `/video_feed?t=${new Date().getTime()}`), 100);
                    $('#start-camera').prop('disabled', false);
                    showToast('Camera stopped successfully', 'success');
                } else {
                    showToast(response.message || 'Failed to stop camera', 'danger');
                    $('#stop-camera').prop('disabled', false);
                }
            })
            .fail(function() {
                showToast('Failed to stop camera', 'danger');
                $('#stop-camera').prop('disabled', false);
            });
    }

    function switchCamera(cameraIndex) {
        if (cameraIndex === currentCameraIndex) return;

        $('.camera-btn').removeClass('active');
        $(`.camera-btn[data-camera="${cameraIndex}"]`).addClass('active');

        const wasRunning = isCameraRunning;
        if (wasRunning) stopCamera();

        $.post('/switch_camera', { camera: cameraIndex })
            .done(function(response) {
                if (response.status === 'success') {
                    currentCameraIndex = cameraIndex;
                    showToast(`Switched to camera ${cameraIndex}`, 'success');
                    if (wasRunning) {
                        setTimeout(startCamera, 500);
                    }
                } else {
                    showToast(response.message || 'Failed to switch camera', 'danger');
                    $(`.camera-btn[data-camera="${currentCameraIndex}"]`).addClass('active');
                    $(`.camera-btn[data-camera="${cameraIndex}"]`).removeClass('active');
                }
            })
            .fail(function() {
                showToast('Failed to switch camera', 'danger');
                $(`.camera-btn[data-camera="${currentCameraIndex}"]`).addClass('active');
                $(`.camera-btn[data-camera="${cameraIndex}"]`).removeClass('active');
            });
    }

    function toggleTracking() {
        const button = $('#toggle-tracking');
        const isCurrentlyEnabled = button.html().includes('Disable');

        $.ajax({
            url: '/toggle_tracking',
            method: 'POST',
            data: { enable: !isCurrentlyEnabled },
            success: function(response) {
                if (response.status === 'success') {
                    const isEnabled = response.tracking_enabled;
                    button.html(`
                        <i class="bi bi-record-circle${isEnabled ? '-fill' : ''} me-1"></i>
                        ${isEnabled ? 'Disable' : 'Enable'} Tracking
                    `);
                    button.toggleClass('btn-success btn-outline-primary', isEnabled);
                    showToast(`Tracking ${isEnabled ? 'enabled' : 'disabled'}`, 'success');
                } else {
                    showToast(response.message || 'Failed to toggle tracking', 'danger');
                }
            },
            error: function() {
                showToast('Failed to toggle tracking', 'danger');
            }
        });
    }

    // Threshold control
    function updateThreshold() {
        $.get('/get_threshold')
            .done(function(data) {
                if (data.status === 'success') {
                    $('#threshold-input').val(data.threshold);
                }
            })
            .fail(function() {
                showToast('Failed to get threshold', 'danger');
            });
    }

    $('#set-threshold').click(function() {
        let threshold = $('#threshold-input').val();
        if (threshold === '' || isNaN(threshold) || threshold < 0) {
            showToast('Please enter a valid threshold value', 'warning');
            return;
        }

        $.post('/set_threshold', { threshold: threshold })
            .done(function(data) {
                if (data.status === 'success') {
                    showToast(data.message, 'success');
                } else {
                    showToast(data.message || 'Failed to set threshold', 'danger');
                }
            })
            .fail(function() {
                showToast('Failed to set threshold', 'danger');
            });
    });

    // File upload handling
    function setupFileUpload() {
        const uploadArea = $('.upload-area');
        const fileInput = $('#file-upload');

        uploadArea.on('dragover', function(e) {
            e.preventDefault();
            $(this).addClass('border-primary bg-light');
        });

        uploadArea.on('dragleave', function(e) {
            e.preventDefault();
            $(this).removeClass('border-primary bg-light');
        });

        uploadArea.on('drop', function(e) {
            e.preventDefault();
            $(this).removeClass('border-primary bg-light');
            if (e.originalEvent.dataTransfer.files.length) {
                fileInput[0].files = e.originalEvent.dataTransfer.files;
                handleFileUpload(fileInput[0].files[0]);
            }
        });

        fileInput.on('change', function() {
            if (this.files.length) {
                handleFileUpload(this.files[0]);
            }
        });

        function handleFileUpload(file) {
            if (!allowedFile(file.name)) {
                showToast('File type not allowed. Please upload images (JPG/PNG) or videos (MP4/AVI/MOV).', 'danger');
                $('#upload-results').html(`
                    <div class="alert alert-danger">
                        File type not allowed. Please upload images (JPG/PNG) or videos (MP4/AVI/MOV).
                    </div>
                `);
                return;
            }

            if (file.size > 100 * 1024 * 1024) {
                showToast('File too large (max 100MB).', 'danger');
                $('#upload-results').html(`
                    <div class="alert alert-danger">
                        File too large (max 100MB).
                    </div>
                `);
                return;
            }

            const formData = new FormData();
            formData.append('file', file);

            $('#upload-results').html(`
                <div class="text-center py-4">
                    <div class="spinner-border text-primary mb-3" role="status">
                        <span class="visually-hidden">Loading...</span>
                    </div>
                    <p>Processing ${file.name}...</p>
                    <div class="progress mt-3">
                        <div class="progress-bar progress-bar-striped progress-bar-animated" style="width: 0%"></div>
                    </div>
                </div>
            `);

            $.ajax({
                url: '/upload',
                type: 'POST',
                data: formData,
                processData: false,
                contentType: false,
                xhr: function() {
                    const xhr = new window.XMLHttpRequest();
                    xhr.upload.addEventListener('progress', function(e) {
                        if (e.lengthComputable) {
                            const percentComplete = Math.round((e.loaded / e.total) * 100);
                            $('#upload-results .progress-bar').css('width', percentComplete + '%').text(percentComplete + '%');
                        }
                    }, false);
                    return xhr;
                },
                success: function(response) {
                    if (response.status === 'success') {
                        let mediaHtml = '';
                        if (response.type === 'image') {
                            mediaHtml = `<img src="${response.processed_url}" alt="Processed Image" class="img-fluid rounded">`;
                        } else if (response.type === 'video') {
                            mediaHtml = `
                                <video controls class="img-fluid rounded">
                                    <source src="${response.processed_url}" type="video/mp4">
                                    Your browser does not support the video tag.
                                </video>
                            `;
                        }

                        const countsHtml = `
                            <h6 class="mt-3">Detection Counts:</h6>
                            <ul class="list-group">
                                <li class="list-group-item">Cars: ${response.counts.car || 0}</li>
                                <li class="list-group-item">Trucks: ${response.counts.truck || 0}</li>
                                <li class="list-group-item">Buses: ${response.counts.bus || 0}</li>
                                <li class="list-group-item">Motorcycles: ${response.counts.motorcycle || 0}</li>
                                <li class="list-group-item">Total: ${Object.values(response.counts).reduce((a, b) => a + b, 0)}</li>
                            </ul>
                        `;

                        $('#upload-results').html(`
                            <div class="text-center">
                                ${mediaHtml}
                                ${countsHtml}
                            </div>
                        `);
                        showToast(response.message, 'success');
                        updateStats();
                        updateDetectionLog();
                        loadHistory();
                    } else {
                        $('#upload-results').html(`
                            <div class="alert alert-danger">
                                ${response.message || 'Failed to process file'}
                            </div>
                        `);
                        showToast(response.message || 'Failed to process file', 'danger');
                    }
                },
                error: function() {
                    $('#upload-results').html(`
                        <div class="alert alert-danger">
                            Failed to upload file
                        </div>
                    `);
                    showToast('Failed to upload file', 'danger');
                }
            });
        }
    }

    // Reset counts
    $('#reset-counts').click(function() {
        $.get('/reset_counts')
            .done(function(response) {
                if (response.status === 'success') {
                    updateStats();
                    showToast('Counts reset successfully', 'success');
                } else {
                    showToast(response.message || 'Failed to reset counts', 'danger');
                }
            })
            .fail(function() {
                showToast('Failed to reset counts', 'danger');
            });
    });

    // Refresh buttons
    $('#refresh-history').click(function() {
        loadHistory();
        showToast('History refreshed', 'success');
    });

    $('#refresh-radar').click(function() {
        loadRadarChart();
        showToast('Radar chart refreshed', 'success');
    });

    $('#refresh-line').click(function() {
        loadLineChart();
        showToast('Line chart refreshed', 'success');
    });

    $('#refresh-bar').click(function() {
        loadBarChart();
        showToast('Bar chart refreshed', 'success');
    });

    $('#refresh-alerts').click(function() {
        loadAlerts();
        showToast('Alerts refreshed', 'success');
    });

    // Camera button events
    $('.camera-btn').click(function() {
        const cameraIndex = parseInt($(this).data('camera'));
        switchCamera(cameraIndex);
    });

    // Camera control buttons
    $('#start-camera').click(startCamera);
    $('#stop-camera').click(stopCamera);
    $('#toggle-tracking').click(toggleTracking);

    // Initialize
    setupFileUpload();
    updateStats();
    loadHistory();
    updateDetectionLog();
    loadRadarChart();
    loadLineChart();
    loadBarChart();
    loadAlerts();
    updateThreshold();

    // Periodic updates
    setInterval(updateStats, 5000);
    setInterval(updateDetectionLog, 5000);
    setInterval(loadHistory, 10000);
    setInterval(loadAlerts, 5000);
});