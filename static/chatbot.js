document.addEventListener("DOMContentLoaded", () => {
    const widget = document.getElementById("chatWidget");
    const toggle = document.getElementById("chatToggle");
    const panel = document.getElementById("chatPanel");
    const closeButton = document.getElementById("chatClose");
    const sendButton = document.getElementById("chatSend");
    const input = document.getElementById("chatInput");
    const messages = document.getElementById("chatMessages");

    if (!widget || !toggle || !panel || !sendButton || !input || !messages) {
        return;
    }

    const appendMessage = (text, type) => {
        const message = document.createElement("div");
        message.className = `chat-message ${type}`;
        message.textContent = text;
        messages.appendChild(message);
        messages.scrollTop = messages.scrollHeight;
    };

    const openPanel = () => {
        panel.hidden = false;
        widget.classList.add("open");
        input.focus();
    };

    const closePanel = () => {
        panel.hidden = true;
        widget.classList.remove("open");
    };

    const sendMessage = async () => {
        const text = input.value.trim();
        if (!text) {
            appendMessage("Please enter a medical question.", "bot");
            return;
        }

        appendMessage(text, "user");
        input.value = "";
        sendButton.disabled = true;
        appendMessage("Checking your medical question...", "bot thinking");

        try {
            const response = await fetch("/chat", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify({ message: text })
            });

            const data = await response.json();
            const thinkingNode = messages.querySelector(".chat-message.thinking:last-child");
            if (thinkingNode) {
                thinkingNode.remove();
            }
            appendMessage(data.reply || "Sorry, something went wrong.", "bot");
        } catch (error) {
            const thinkingNode = messages.querySelector(".chat-message.thinking:last-child");
            if (thinkingNode) {
                thinkingNode.remove();
            }
            appendMessage("Chatbot is unavailable right now. Please try again.", "bot");
        } finally {
            sendButton.disabled = false;
        }
    };

    toggle.addEventListener("click", () => {
        if (panel.hidden) {
            openPanel();
        } else {
            closePanel();
        }
    });

    if (closeButton) {
        closeButton.addEventListener("click", closePanel);
    }

    sendButton.addEventListener("click", sendMessage);
    input.addEventListener("keydown", (event) => {
        if (event.key === "Enter") {
            sendMessage();
        }
    });
});
