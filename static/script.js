document.addEventListener('DOMContentLoaded', () => {
    const videoList = document.getElementById('video-list');
    const addForm = document.getElementById('add-form');
    const commitBtn = document.getElementById('commit-btn');

    // Add new video
    addForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const title = document.getElementById('add-title').value;
        const description = document.getElementById('add-description').value;
        const videoURL = document.getElementById('add-video-url').value;
        const pageURL = document.getElementById('add-page-url').value;

        const response = await fetch('/add', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ Title: title, Description: description, Video_URL: videoURL, Page_URL: pageURL })
        });

        if (response.ok) {
            const video = await response.json();
            addVideoItem(video);
            addForm.reset();
        } else {
            alert('Failed to add video');
        }
    });

    // Edit video
    videoList.addEventListener('click', async (e) => {
        if (e.target.classList.contains('edit-btn')) {
            const videoItem = e.target.closest('.video-item');
            const videoId = videoItem.getAttribute('data-id');
            const title = videoItem.querySelector('h2').textContent;
            const description = videoItem.querySelector('p').textContent;
            const videoURL = videoItem.querySelector('a').getAttribute('href');
            const pageURL = videoItem.querySelectorAll('a')[1].getAttribute('href');

            const newTitle = prompt('Enter new title:', title);
            const newDescription = prompt('Enter new description:', description);
            const newVideoURL = prompt('Enter new video URL:', videoURL);
            const newPageURL = prompt('Enter new page URL:', pageURL);

            if (newTitle && newDescription && newVideoURL && newPageURL) {
                const response = await fetch(`/edit/${videoId}`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({ Title: newTitle, Description: newDescription, Video_URL: newVideoURL, Page_URL: newPageURL })
                });

                if (response.ok) {
                    const video = await response.json();
                    videoItem.querySelector('h2').textContent = video.Title;
                    videoItem.querySelector('p').textContent = video.Description;
                    videoItem.querySelector('a').setAttribute('href', video.Video_URL);
                    videoItem.querySelectorAll('a')[1].setAttribute('href', video.Page_URL);
                } else {
                    alert('Failed to edit video');
                }
            }
        }
    });

    // Delete video
    videoList.addEventListener('click', async (e) => {
        if (e.target.classList.contains('delete-btn')) {
            const videoItem = e.target.closest('.video-item');
            const videoId = videoItem.getAttribute('data-id');

            const response = await fetch(`/delete/${videoId}`, {
                method: 'POST'
            });

            if (response.ok) {
                videoItem.remove();
            } else {
                alert('Failed to delete video');
            }
        }
    });

    // Commit changes
    commitBtn.addEventListener('click', async () => {
        // show loading screen
        document.getElementById('loading-screen').style.display = 'block';
        
        const response = await fetch('/commit', {
            method: 'POST'
        });
        
        // hide loading screen
        document.getElementById('loading-screen').style.display = 'none';
        if (response.ok) {
            alert('Changes committed');
        } else {
            alert('Failed to commit changes. \nPossible reasons: \nYou have not not made any changes, or ¯\\_(ツ)_/¯');
        }
        
    });

    // Helper function to add video item to the list
    function addVideoItem(video) {
        const videoItem = document.createElement('div');
        videoItem.className = 'video-item';
        videoItem.setAttribute('data-id', video.Id);
        videoItem.innerHTML = `
            <h2>${video.Title}</h2>
            <p>${video.Description}</p>
            <a href="${video.Video_URL}" target="_blank">Video Download Link</a>
            <a href="${video.Page_URL}" target="_blank">Watch Video</a>
            <button class="edit-btn">Edit</button>
            <button class="delete-btn">Delete</button>
        `;
        videoList.appendChild(videoItem);
    }
});
