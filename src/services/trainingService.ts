import { toast } from "sonner";
import { getDataset } from "./datasetService";

export interface TrainingParameters {
  model: string;
  epochs: number;
  learningRate: number;
  batchSize: number;
  maxLength: number;
  warmupSteps: number;
  weightDecay: number;
  gradientAccumulationSteps: number;
  useHalfPrecision: boolean;
  useLoRA: boolean;
  saveBestModel: boolean;
  evalSteps: number;
  datasetId: string;
}

export interface TrainingMetrics {
  step: number;
  trainingLoss: number;
  validationLoss: number;
}

export interface EvaluationMetric {
  metric: string;
  value: number;
  previous: number;
}

export interface TrainingState {
  id: string;
  status: 'idle' | 'preparing' | 'training' | 'paused' | 'completed' | 'error';
  progress: number;
  currentStep: number;
  totalSteps: number;
  metrics: TrainingMetrics[];
  evaluationMetrics: EvaluationMetric[];
  elapsedTime: number;
  estimatedTimeRemaining: number;
  error?: string;
  params: TrainingParameters;
  logs: string[];
}

// In-memory storage for training jobs
let trainingJobs: TrainingState[] = [];

// This is a mock function to simulate the actual training process
let trainingInterval: ReturnType<typeof setInterval> | null = null;

export const startTraining = async (params: TrainingParameters): Promise<TrainingState> => {
  try {
    // Validate parameters
    if (!params.datasetId) {
      throw new Error("No dataset selected");
    }
    
    // Get dataset from backend
    const dataset = await getDataset(params.datasetId);
    if (!dataset) {
      throw new Error("Selected dataset not found");
    }
    
    if (dataset.status !== 'success') {
      throw new Error("Dataset is not ready for training");
    }
    
    // Calculate total steps based on epochs and dataset size
    const datasetSize = dataset.num_samples; // Use num_samples from backend response
    const stepsPerEpoch = Math.ceil(datasetSize / params.batchSize);
    const totalSteps = stepsPerEpoch * params.epochs;
    
    // Create training job
    const trainingJob: TrainingState = {
      id: Math.random().toString(36).substring(2, 9),
      status: 'preparing',
      progress: 0,
      currentStep: 0,
      totalSteps,
      metrics: [],
      evaluationMetrics: [
        { metric: "Accuracy", value: 0, previous: 0 },
        { metric: "F1 Score", value: 0, previous: 0 },
        { metric: "Precision", value: 0, previous: 0 },
        { metric: "Recall", value: 0, previous: 0 }
      ],
      elapsedTime: 0,
      estimatedTimeRemaining: totalSteps * 2, // Rough estimate: 2 seconds per step
      params,
      logs: [
        `INFO: Starting fine-tuning of ${params.model} model`,
        `INFO: Loading dataset with ${datasetSize} examples`,
        `INFO: Training with batch size ${params.batchSize}, learning rate ${params.learningRate.toExponential(2)}`
      ]
    };
    
    // Add to training jobs collection
    trainingJobs.push(trainingJob);
    
    // Start the mock training process
    setTimeout(() => {
      trainingJob.status = 'training';
      updateTrainingJob(trainingJob);
      
      // Simulate training steps
      let lastLogStep = 0;
      let timeCounter = 0;
      
      trainingInterval = setInterval(() => {
        timeCounter += 1;
        trainingJob.elapsedTime = timeCounter;
        
        // Only update step every 2 seconds to simulate computation time
        if (timeCounter % 2 === 0) {
          trainingJob.currentStep += 1;
        }
        
        // Update progress percentage
        trainingJob.progress = Math.min(100, (trainingJob.currentStep / trainingJob.totalSteps) * 100);
        
        // Update estimated time remaining
        const timePerStep = timeCounter / trainingJob.currentStep;
        trainingJob.estimatedTimeRemaining = Math.max(0, 
          (trainingJob.totalSteps - trainingJob.currentStep) * timePerStep
        );
        
        // Generate training metrics
        if (trainingJob.currentStep % 10 === 0) {
          // Generate decreasing loss values to simulate learning
          const baseLoss = Math.max(0.2, 2.0 - (trainingJob.currentStep / trainingJob.totalSteps) * 1.8);
          const trainingLoss = baseLoss - 0.05 + (Math.random() * 0.1);
          const validationLoss = trainingLoss + 0.1 + (Math.random() * 0.2);
          
          trainingJob.metrics.push({
            step: trainingJob.currentStep,
            trainingLoss: parseFloat(trainingLoss.toFixed(4)),
            validationLoss: parseFloat(validationLoss.toFixed(4))
          });
        }
        
        // Add logs periodically
        if (trainingJob.currentStep - lastLogStep >= 10) {
          lastLogStep = trainingJob.currentStep;
          const metric = trainingJob.metrics[trainingJob.metrics.length - 1];
          
          trainingJob.logs.push(
            `INFO: Step ${trainingJob.currentStep}/${trainingJob.totalSteps}: ` +
            `Training loss: ${metric?.trainingLoss.toFixed(3)}, ` +
            `Validation loss: ${metric?.validationLoss.toFixed(3)}`
          );
        }
        
        // Evaluation metrics every 100 steps
        if (trainingJob.currentStep % 100 === 0) {
          trainingJob.logs.push(`INFO: Saving checkpoint at step ${trainingJob.currentStep}`);
          trainingJob.logs.push(`INFO: Evaluating model on validation set`);
          
          // Update evaluation metrics with improving scores
          const improvement = (trainingJob.currentStep / trainingJob.totalSteps) * 0.3;
          
          trainingJob.evaluationMetrics = trainingJob.evaluationMetrics.map(metric => {
            const previous = metric.value;
            const newValue = Math.min(0.95, 0.6 + improvement + (Math.random() * 0.1));
            
            return {
              ...metric,
              value: parseFloat(newValue.toFixed(2)),
              previous: previous
            };
          });
          
          trainingJob.logs.push(
            `INFO: Validation metrics: Accuracy: ${trainingJob.evaluationMetrics[0].value.toFixed(2)}, ` +
            `F1: ${trainingJob.evaluationMetrics[1].value.toFixed(2)}`
          );
        }
        
        // Complete training when done
        if (trainingJob.currentStep >= trainingJob.totalSteps) {
          trainingJob.status = 'completed';
          trainingJob.progress = 100;
          trainingJob.logs.push(`INFO: Training completed successfully`);
          trainingJob.logs.push(`INFO: Final metrics - Accuracy: ${trainingJob.evaluationMetrics[0].value.toFixed(2)}, F1: ${trainingJob.evaluationMetrics[1].value.toFixed(2)}`);
          
          clearInterval(trainingInterval!);
          trainingInterval = null;
          
          toast.success("Training completed successfully");
        }
        
        updateTrainingJob(trainingJob);
      }, 1000);
    }, 2000);
    
    toast.success("Training job created", {
      description: `Fine-tuning ${params.model} for ${params.epochs} epochs`
    });
    
    return trainingJob;
  } catch (error) {
    toast.error("Failed to start training", {
      description: error.message
    });
    throw error;
  }
};

// Rest of the functions remain the same...
export const pauseTraining = (id: string): TrainingState | undefined => {
  const job = getTrainingJob(id);
  if (!job) return undefined;
  
  if (job.status === 'training') {
    job.status = 'paused';
    updateTrainingJob(job);
    
    if (trainingInterval) {
      clearInterval(trainingInterval);
      trainingInterval = null;
    }
    
    toast("Training paused");
    return job;
  }
  
  return job;
};

export const resumeTraining = (id: string): TrainingState | undefined => {
  const job = getTrainingJob(id);
  if (!job || job.status !== 'paused') return undefined;
  
  job.status = 'training';
  updateTrainingJob(job);
  
  // Resume the mock training process
  let timeCounter = job.elapsedTime;
  let lastLogStep = job.currentStep;
  
  trainingInterval = setInterval(() => {
    timeCounter += 1;
    job.elapsedTime = timeCounter;
    
    if (timeCounter % 2 === 0) {
      job.currentStep += 1;
    }
    
    job.progress = Math.min(100, (job.currentStep / job.totalSteps) * 100);
    
    const timePerStep = timeCounter / job.currentStep;
    job.estimatedTimeRemaining = Math.max(0, 
      (job.totalSteps - job.currentStep) * timePerStep
    );
    
    if (job.currentStep % 10 === 0) {
      const baseLoss = Math.max(0.2, 2.0 - (job.currentStep / job.totalSteps) * 1.8);
      const trainingLoss = baseLoss - 0.05 + (Math.random() * 0.1);
      const validationLoss = trainingLoss + 0.1 + (Math.random() * 0.2);
      
      job.metrics.push({
        step: job.currentStep,
        trainingLoss: parseFloat(trainingLoss.toFixed(4)),
        validationLoss: parseFloat(validationLoss.toFixed(4))
      });
    }
    
    if (job.currentStep - lastLogStep >= 10) {
      lastLogStep = job.currentStep;
      const metric = job.metrics[job.metrics.length - 1];
      
      job.logs.push(
        `INFO: Step ${job.currentStep}/${job.totalSteps}: ` +
        `Training loss: ${metric?.trainingLoss.toFixed(3)}, ` +
        `Validation loss: ${metric?.validationLoss.toFixed(3)}`
      );
    }
    
    if (job.currentStep >= job.totalSteps) {
      job.status = 'completed';
      job.progress = 100;
      job.logs.push(`INFO: Training completed successfully`);
      
      clearInterval(trainingInterval!);
      trainingInterval = null;
      
      toast.success("Training completed successfully");
    }
    
    updateTrainingJob(job);
  }, 1000);
  
  toast("Training resumed");
  return job;
};

export const stopTraining = (id: string): TrainingState | undefined => {
  const job = getTrainingJob(id);
  if (!job) return undefined;
  
  if (job.status === 'training' || job.status === 'paused') {
    job.status = 'completed';
    job.logs.push(`INFO: Training stopped by user at step ${job.currentStep}`);
    
    if (trainingInterval) {
      clearInterval(trainingInterval);
      trainingInterval = null;
    }
    
    toast.warning("Training stopped", {
      description: "Your model progress has been saved."
    });
    
    return updateTrainingJob(job);
  }
  
  return job;
};

// Get all training jobs
export const getTrainingJobs = (): TrainingState[] => {
  return [...trainingJobs];
};

// Get a specific training job by ID
export const getTrainingJob = (id: string): TrainingState | undefined => {
  return trainingJobs.find(job => job.id === id);
};

// Get the latest active training job
export const getLatestTrainingJob = (): TrainingState | undefined => {
  const activeJobs = trainingJobs.filter(
    job => job.status === 'training' || job.status === 'paused'
  );
  
  if (activeJobs.length > 0) {
    return activeJobs[activeJobs.length - 1];
  }
  
  // Return the latest completed job if no active jobs
  return trainingJobs[trainingJobs.length - 1];
};

// Update a training job
export const updateTrainingJob = (updatedJob: TrainingState): TrainingState => {
  trainingJobs = trainingJobs.map(job => 
    job.id === updatedJob.id ? updatedJob : job
  );
  return updatedJob;
};

// Format time in HH:MM:SS
export const formatTime = (seconds: number): string => {
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  const secs = Math.floor(seconds % 60);
  
  return [
    hours.toString().padStart(2, '0'),
    minutes.toString().padStart(2, '0'),
    secs.toString().padStart(2, '0')
  ].join(':');
};