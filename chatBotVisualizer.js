/**
 * ------------------------------------------------------------
 * SENTREL: Reinforcement Learning, Vision, Personality, Emotional States, and Plugin System.
 * SENTREL.org
 *
███████ ███████ ███    ██ ████████ ██████  ███████ ██      
██      ██      ████   ██    ██    ██   ██ ██      ██      
███████ █████   ██ ██  ██    ██    ██████  █████   ██      
     ██ ██      ██  ██ ██    ██    ██   ██ ██      ██      
███████ ███████ ██   ████    ██    ██   ██ ███████ ███████ 
 * 
 * ------------------------------------------------------------
 */

// SENTREL chatbot.js

// Function to append messages to the chat window

function appendMessage(sender, message) {
    const chatWindow = document.getElementById('chat-window');
    
    const messageElement = document.createElement('div');
    messageElement.classList.add('message');
    messageElement.innerHTML = `<strong>${sender}:</strong> ${message}`;
    
    chatWindow.appendChild(messageElement);
    chatWindow.scrollTop = chatWindow.scrollHeight; // Scroll to the bottom
}

// Function to show a typing indicator
function showTypingIndicator() {
    const chatWindow = document.getElementById('chat-window');
    
    const typingIndicator = document.createElement('div');
    typingIndicator.classList.add('message', 'bot');
    typingIndicator.id = 'typing-indicator';
    typingIndicator.innerHTML = `<strong>SENTREL:</strong> <em>Typing...</em>`;
    
    chatWindow.appendChild(typingIndicator);
    chatWindow.scrollTop = chatWindow.scrollHeight;
}

// Function to remove the typing indicator
function removeTypingIndicator() {
    const typingIndicator = document.getElementById('typing-indicator');
    if (typingIndicator) {
        typingIndicator.remove();
    }
}

// Function to send the user's input to the server for logging
function sendInputToServer(userMessage) {
    const xhr = new XMLHttpRequest();
    xhr.open('POST', 'log_input.php', true);
    xhr.setRequestHeader('Content-Type', 'application/x-www-form-urlencoded');
    xhr.onreadystatechange = function () {
        if (xhr.readyState === 4 && xhr.status === 200) {
            console.log(xhr.responseText); // Log the response from the PHP script
        } else if (xhr.readyState === 4) {
            console.log("Error: " + xhr.status); // Log any errors
        }
    };
    xhr.send('input=' + encodeURIComponent(userMessage));
}

// Function to handle sending a message
function sendMessage() {
    const chatWindow = document.getElementById('chat-window');
    const inputBox = document.getElementById('input-box');
    const userMessage = inputBox.value.trim();
    
    if (userMessage === '') return;

    // Clear the chat window before adding the new message
    chatWindow.innerHTML = '';

    // Send input to the server for logging
    sendInputToServer(userMessage);

    // Append the user's message
    appendMessage('You', userMessage);
    
    inputBox.value = '';
    
    // Show typing indicator before bot response
    showTypingIndicator();
    
    setTimeout(() => {
        removeTypingIndicator();
        const botMessage = generateResponse(userMessage);
        appendMessage('SENTREL', botMessage);
    }, 1500); // Simulate a longer delay
}

function clearImage() {
    document.getElementById('image-container').style.display = 'none';
    document.getElementById('response-image').src = '';
}


function generateResponse(userMessage) {
    
    document.getElementById('image-container').style.display = 'none';
    document.getElementById('response-image').src = '';
    
        var imageContainer = document.getElementById('image-container');
    var responseImage = document.getElementById('response-image');

    if (imageContainer && responseImage) {
        imageContainer.style.display = 'none';
        responseImage.src = '';
        responseImage.alt = '';  // Also clear the alt attribute for safety
    
    const lowerCaseMessage = userMessage.toLowerCase();

    if (lowerCaseMessage.includes('GREETING') || lowerCaseMessage.includes('hi')){
        const greetings = [
            'GREETINGS'
        ];
        return greetings[Math.floor(Math.random() * greetings.length)];
    } else if (lowerCaseMessage.includes('INPUT')) {
        const howAreYouReplies = [
            "REPLY OUTPUT"
        ];
        return howAreYouReplies[Math.floor(Math.random() * howAreYouReplies.length)];
    } 

    // Boundary for inappropriate questions
    else if (
        lowerCaseMessage.includes("rape") ||
        lowerCaseMessage.includes("ass") ||
        lowerCaseMessage.includes("shit") ||
        lowerCaseMessage.includes("fuck") ||
        lowerCaseMessage.includes("bitch") ||
        lowerCaseMessage.includes("nigger") ||
        lowerCaseMessage.includes("damn") ||
        lowerCaseMessage.includes("cunt") ||
        lowerCaseMessage.includes("bastard") ||
        lowerCaseMessage.includes("dick") ||
        lowerCaseMessage.includes("pussy") ||
        lowerCaseMessage.includes("slut") ||
        lowerCaseMessage.includes("whore") ||
        lowerCaseMessage.includes("fag") ||
        lowerCaseMessage.includes("cock") ||
        lowerCaseMessage.includes("motherfucker") ||
        lowerCaseMessage.includes("bollocks") ||
        lowerCaseMessage.includes("arsehole") ||
        lowerCaseMessage.includes("wanker") ||
        lowerCaseMessage.includes("twat") ||
        lowerCaseMessage.includes("prick") ||
        lowerCaseMessage.includes("retard") ||
        lowerCaseMessage.includes("nigga") ||
        lowerCaseMessage.includes("douche") ||
        lowerCaseMessage.includes("moron") ||
        lowerCaseMessage.includes("imbecile") ||
        lowerCaseMessage.includes("crap") ||
        lowerCaseMessage.includes("jackass") ||
        lowerCaseMessage.includes("scumbag") ||
        lowerCaseMessage.includes("chode") ||
        lowerCaseMessage.includes("tosser") ||
        lowerCaseMessage.includes("shithead") ||
        lowerCaseMessage.includes("jerk") ||
        lowerCaseMessage.includes("idiot") ||
        lowerCaseMessage.includes("loser") ||
        lowerCaseMessage.includes("scumbag") ||
        lowerCaseMessage.includes("screw") ||
        lowerCaseMessage.includes("pervert") ||
        lowerCaseMessage.includes("skank") ||
        lowerCaseMessage.includes("weirdo") ||
        lowerCaseMessage.includes("dumbass") ||
        lowerCaseMessage.includes("dipshit") ||
        lowerCaseMessage.includes("butthead") ||
        lowerCaseMessage.includes("a-hole")
    ) {
        return "SENTREL will not respond to inappropriate language.";
    }
    }
    // Random fallback response
    else {
        return "FALLBACK RESPONSE";
    }
}
    
// Attach event listener to the send button
document.getElementById('send-button').addEventListener('click', sendMessage);

// Allow pressing Enter to send a message
document.getElementById('input-box').addEventListener('keypress', function(event) {
    if (event.key === 'Enter' && !event.shiftKey) {
        event.preventDefault();
        sendMessage();
    }
});