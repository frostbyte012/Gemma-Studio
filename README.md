# Gemma Studio

Gemma Studio is a web-based application designed to streamline dataset management, model training, and deployment workflows. Built with modern technologies like **React**, **TypeScript**, **Vite**, and **Tailwind CSS**, it provides an intuitive interface for managing datasets, configuring training parameters, monitoring training progress, and exporting trained models.

# Inspiration
Inspired by GSoC 25 from Google DeepMind, the goal is to create a user-friendly Gemma Model Fine-tuning UI using tools like Streamlit or Gradio. The UI enables users to:

- **Dataset Uploading**: Support various formats (CSV, JSONL, text files) with validation, preprocessing, and optional data augmentation.
- **Hyperparameter Configuration**: Adjust key parameters (learning rate, batch size, epochs) with sensible defaults and tooltips.
- **Training Progress Visualization**: Display real-time metrics (loss curves, accuracy, F1-score) and examples of generated text.
- **Model Download/Export**: Export fine-tuned models in formats like TensorFlow SavedModel, PyTorch, or GGUF.
- **Cloud Integration**: Optionally integrate with Google Cloud Storage and Vertex AI for scalable training and data storage.
- **Documentation**: Provide clear documentation and step-by-step examples for ease of use.

---

## Project Structure

The project is organized as follows:

```
gemma-studio/
├── public/                      # Static assets
├── src/
│   ├── components/              # React components
│   │   ├── dashboard/           # Dashboard components
│   │   │   └── WelcomeCard.tsx  # Welcome component on dashboard
│   │   ├── dataset/             # Dataset related components
│   │   │   ├── DatasetPreview.tsx  # Preview uploaded datasets
│   │   │   └── DatasetUpload.tsx   # Upload datasets UI
│   │   ├── layout/              # Layout components
│   │   │   ├── Layout.tsx       # Main layout wrapper
│   │   │   └── Navbar.tsx       # Navigation bar
│   │   ├── models/              # Model components
│   │   │   └── ModelExport.tsx  # Export trained models
│   │   ├── training/            # Training components
│   │   │   ├── HyperparameterConfig.tsx  # Configure training parameters
│   │   │   └── TrainingProgress.tsx      # Training progress visualization
│   │   └── ui/                  # shadcn/ui components
│   │       └── ...              # Various UI components (buttons, cards, etc.)
│   ├── hooks/                   # Custom React hooks
│   │   ├── use-mobile.tsx       # Hook for responsive design
│   │   └── use-toast.ts         # Toast notification hook
│   ├── lib/                     # Utility functions
│   │   ├── animations.ts        # Animation utilities
│   │   └── utils.ts             # General utilities
│   ├── pages/                   # Page components
│   │   ├── Dashboard.tsx        # Dashboard page
│   │   ├── Datasets.tsx         # Datasets management page
│   │   ├── Index.tsx            # Landing page
│   │   ├── Models.tsx           # Model export page
│   │   ├── NotFound.tsx         # 404 page
│   │   ├── Settings.tsx         # Settings page
│   │   └── Training.tsx         # Training configuration and monitoring page
│   ├── services/                # Backend service integrations
│   │   ├── datasetService.ts    # Dataset management functionality
│   │   ├── modelService.ts      # Model export functionality
│   │   └── trainingService.ts   # Training functionality
│   ├── App.css                  # App-wide styles
│   ├── App.tsx                  # Main application component with routing
│   ├── index.css                # Global styles
│   ├── main.tsx                 # Application entry point
│   └── vite-env.d.ts            # Vite environment types
├── eslint.config.js             # ESLint configuration
├── tailwind.config.ts           # Tailwind CSS configuration
├── tsconfig.json                # TypeScript configuration
└── vite.config.ts               # Vite configuration
```

---

## Technologies Used

- **Vite**: Fast build tool for modern web projects.
- **TypeScript**: Strongly typed JavaScript for better code quality.
- **React**: Component-based UI library.
- **shadcn-ui**: Pre-built UI components.
- **Tailwind CSS**: Utility-first CSS framework.

---

## Installation and Usage

### Prerequisites
- **Node.js** and **npm** installed on your system. You can install them using [nvm](https://github.com/nvm-sh/nvm#installing-and-updating).

### Steps to Install and Run Locally
1. Clone the repository:
   ```bash
   git clone <YOUR_GIT_URL>
   ```

2. Navigate to the project directory:
   ```bash
   cd gemma-studio
   ```

3. Install dependencies:
   ```bash
   npm install
   ```

4. Start the development server:
   ```bash
   npm run dev
   ```

5. Open your browser and navigate to `http://localhost:3000` to view the application.

---

## Deployment

### Deploying to Hugging Face Spaces
1. Package the application as a Docker container:
   ```bash
   docker build -t gemma-studio .
   ```

2. Push the Docker image to Hugging Face Spaces:
   - Follow the [Hugging Face Spaces Docker documentation](https://huggingface.co/docs/hub/spaces-docker) to deploy your container.

### Deploying to Google Cloud Run
1. Authenticate with Google Cloud:
   ```bash
   gcloud auth login
   gcloud auth configure-docker
   ```

2. Build and push the Docker image:
   ```bash
   docker build -t gcr.io/<YOUR_PROJECT_ID>/gemma-studio .
   docker push gcr.io/<YOUR_PROJECT_ID>/gemma-studio
   ```

3. Deploy to Cloud Run:
   ```bash
   gcloud run deploy gemma-studio \
       --image gcr.io/<YOUR_PROJECT_ID>/gemma-studio \
       --platform managed \
       --region us-central1 \
       --allow-unauthenticated
   ```

### Integrating with Vertex AI
- Use Vertex AI for advanced model training and deployment.
- Integrate the backend services (`trainingService.ts`, `modelService.ts`) with Vertex AI APIs for seamless training and deployment workflows.
- Refer to the [Vertex AI documentation](https://cloud.google.com/vertex-ai/docs) for more details.

---

## Future Scope

1. **Hugging Face Integration**:
   - Deploy models directly to Hugging Face Spaces for easy sharing and inference.

2. **Google Cloud Integration**:
   - Use Google Cloud Storage for dataset management.
   - Leverage Vertex AI for scalable model training and deployment.

3. **Custom Domain Support**:
   - Integrate with platforms like Netlify or Vercel for hosting under a custom domain.

4. **Enhanced UI/UX**:
   - Add more interactive visualizations for training progress and dataset insights.

5. **Multi-Cloud Support**:
   - Extend deployment options to AWS and Azure.

---

## Contributing

Contributions are welcome! Please follow these steps:
1. Fork the repository.
2. Create a new branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. Commit your changes:
   ```bash
   git commit -m "Add your message here"
   ```
4. Push to your branch:
   ```bash
   git push origin feature/your-feature-name
   ```
5. Open a pull request.

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## Contact

For any questions or feedback, feel free to reach out at [ai.frostbyte012@gmail.com].