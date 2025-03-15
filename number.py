import pygame
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Initialize Pygame
pygame.init()

# Screen dimensions (set to full screen)
WIDTH, HEIGHT = 900, 900
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Digit Recognition Neural Network")

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (128, 128, 128)
RED = (255, 0, 0)

# Load and preprocess MNIST data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255
x_train = x_train.reshape((-1, 28 * 28))
x_test = x_test.reshape((-1, 28 * 28))

# Create and train a simple neural network
model = Sequential([
    Dense(128, activation='relu', input_shape=(784,)),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5, batch_size=32, validation_split=0.2)

# Force the model to be called with a dummy input so that its input is defined.
dummy_input = np.zeros((1, 784), dtype=np.float32)
_ = model(dummy_input)

# Create an intermediate model to extract activations.
activations = [layer.output for layer in model.layers]
intermediate_model = tf.keras.Model(inputs=model.inputs, outputs=activations)

# Drawing canvas (28x28 pixels scaled up)
canvas_size = 280  # 28x28 scaled by 10
# Initialize the canvas with ones (white background)
canvas = np.ones((28, 28))
scale = canvas_size // 28

# Neural network visualization parameters
layer_sizes = [128, 64, 10]           # Actual sizes of each layer
display_layer_sizes = [16, 8, 10]       # Reduced number to display (first and second layers)
node_radius = 10
layer_spacing = 200
start_x = canvas_size + 50
start_y = 50

# Main game loop variables
running = True
drawing = False

def draw_network(screen, activations):
    # Compute positions for displayed neurons using the reduced display sizes.
    layer_positions = []
    for layer_idx, (full_size, disp_size, activation) in enumerate(zip(layer_sizes, display_layer_sizes, activations)):
        positions = []
        x = start_x + layer_idx * layer_spacing
        # Sample indices evenly if the full layer is larger than the display count.
        if full_size > disp_size:
            indices = np.linspace(0, full_size - 1, disp_size, dtype=int)
        else:
            indices = np.arange(full_size)
        # Compute vertical positions evenly spaced.
        for node_idx in range(len(indices)):
            if disp_size > 1:
                y = start_y + node_idx * (HEIGHT - 100) // (disp_size - 1)
            else:
                y = HEIGHT // 2
            positions.append((x, y))
        layer_positions.append((positions, indices))
        
    # Draw connections between layers.
    for i in range(len(layer_positions) - 1):
        positions, _ = layer_positions[i]
        positions_next, _ = layer_positions[i+1]
        for pos in positions:
            for pos_next in positions_next:
                pygame.draw.line(screen, GRAY, pos, pos_next, 1)
    
    # Draw neurons.
    for layer_idx, ((positions, indices), activation) in enumerate(zip(layer_positions, activations)):
        for display_idx, pos in enumerate(positions):
            neuron_index = indices[display_idx]
            brightness = int(min(activation[0][neuron_index] * 255, 255))
            color = (brightness, brightness, 0)  # Yellow tint based on activation.
            pygame.draw.circle(screen, color, pos, node_radius)
            pygame.draw.circle(screen, WHITE, pos, node_radius, 1)

def predict_digit(canvas):
    # Invert the canvas so that black drawing on white becomes white digit on black (as in MNIST)
    input_data = (1 - canvas).reshape(1, 784)
    prediction = model.predict(input_data, verbose=0)
    layer_outputs = intermediate_model.predict(input_data, verbose=0)
    return np.argmax(prediction), layer_outputs

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            drawing = True
        elif event.type == pygame.MOUSEBUTTONUP:
            drawing = False
        elif event.type == pygame.MOUSEMOTION and drawing:
            x, y = event.pos
            if 0 <= x < canvas_size and 0 <= y < canvas_size:
                canvas_x, canvas_y = x // scale, y // scale
                canvas[canvas_y, canvas_x] = 0.0  # Draw black on white background.
        elif event.type == pygame.KEYDOWN and event.key == pygame.K_c:
            canvas.fill(1)  # Clear canvas (reset to white)

    # Draw the canvas.
    pygame.draw.rect(screen, WHITE, (0, 0, canvas_size, canvas_size))
    for y in range(28):
        for x in range(28):
            if canvas[y, x] < 1:
                pygame.draw.rect(screen, BLACK, (x * scale, y * scale, scale, scale))

    # Predict the digit and draw the network visualization.
    digit, layer_activations = predict_digit(canvas)
    draw_network(screen, layer_activations)

    font = pygame.font.Font(None, 36)
    text = font.render(f"Predicted Digit: {digit}", True, RED)
    screen.blit(text, (canvas_size + 10, HEIGHT - 50))

    pygame.display.flip()

pygame.quit()
