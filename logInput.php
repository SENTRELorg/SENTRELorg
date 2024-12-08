<?php
if ($_SERVER["REQUEST_METHOD"] == "POST" && isset($_POST['input'])) {
    $input = $_POST['input'];
    $file = 'user_inputs.txt';

    // Append the input to the file with a newline
    file_put_contents($file, $input . PHP_EOL, FILE_APPEND);

    // Send a response back to JavaScript
    echo "SENTREL Input saved successfully.";
} else {
    echo "No input received.";
}
?>