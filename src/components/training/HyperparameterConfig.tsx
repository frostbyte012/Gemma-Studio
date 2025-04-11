import React, { useState, useEffect} from "react"; // Added useState import
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Slider } from "@/components/ui/slider";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";
import { Switch } from "@/components/ui/switch";
import { HelpCircle, Zap } from "lucide-react";
import { useAnimateOnMount } from "@/lib/animations";
import { toast } from "sonner";
import { useNavigate } from "react-router-dom";
import { startTraining } from "@/services/trainingService";
import { getDatasets } from "@/services/datasetService";

// Define the type for params
interface Params {
  model: string;
  epochs: number;
  learningRate: number;
  batchSize: number;
  maxLength: number;
  warmupSteps: number;
  weightDecay: number;
  gradientAccumulationSteps: number;
}


export function HyperparameterConfig() {
  const navigate = useNavigate();
  const [params, setParams] = useState({
    // Basic params
    model: "gemma-7b",
    epochs: 3,
    learningRate: 2e-5,
    batchSize: 8,
    
    // Advanced params
    maxLength: 512,
    warmupSteps: 100,
    weightDecay: 0.01,
    gradientAccumulationSteps: 1,
    
    // Training settings
    useHalfPrecision: true,
    useLoRA: true,
    saveBestModel: true,
    evalSteps: 100,
    
    // Selected dataset
    datasetId: ""
  });
  
  const { styles } = useAnimateOnMount({
    type: 'slide',
    direction: 'up',
    duration: 500
  });
  
  const handleBasicParamChange = (key: string, value: any) => {
    setParams(prev => ({ ...prev, [key]: value }));
  };
  
  const handleResetToDefaults = () => {
    setParams({
      model: "gemma-7b",
      epochs: 3,
      learningRate: 2e-5,
      batchSize: 8,
      maxLength: 512,
      warmupSteps: 100,
      weightDecay: 0.01,
      gradientAccumulationSteps: 1,
      useHalfPrecision: true,
      useLoRA: true,
      saveBestModel: true,
      evalSteps: 100,
      datasetId: params.datasetId // Keep the selected dataset
    });
    toast.success("Parameters reset to defaults");
  };
  
const handleStartTraining = async () => {  // Changed to async
  try {
    // Get datasets and validate we have at least one successful dataset
    const datasets = await getDatasets(); // Added await
    const successfulDatasets = datasets.filter((d: any) => d.status === 'success'); // Added type assertion
    
    if (successfulDatasets.length === 0) {
      toast.error("No valid datasets", {
        description: "Please upload and validate at least one dataset before training"
      });
      return;
    }
    
    // Use the first successful dataset if none is selected
    const datasetId = params.datasetId || successfulDatasets[0].id;
    
    // Start training with the service
    const trainingJob = await startTraining({ // Added await
      ...params,
      datasetId
    });
    
    if (!trainingJob?.id) { // Added null check
      throw new Error("Training job ID not received");
    }
    
    // Switch to the progress tab
    navigate("/training", { 
      state: { 
        activeTab: "progress", 
        jobId: trainingJob.id.toString() // Ensure string type
      } 
    });
  } catch (error) {
    toast.error("Failed to start training", {
      description: error instanceof Error ? error.message : "Unknown error"
    });
  }
};

// // Update datasets getter to be consistent
// const datasets = (getDatasets() ?? []).filter(d => d?.status === 'success');
const [datasets, setDatasets] = useState<any[]>([]);

useEffect(() => {
  async function fetchDatasets() {
    try {
      const data = await getDatasets();
      setDatasets((data ?? []).filter(d => d?.status === "success"));
    } catch (error) {
      console.error("Error fetching datasets:", error);
    }
  }

  fetchDatasets();
}, []);

  return (
    <TooltipProvider>
      <Card className="border shadow-sm" style={styles}>
        <CardHeader>
          <CardTitle>Hyperparameter Configuration</CardTitle>
          <CardDescription>
            Configure the parameters for fine-tuning your model
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          <Tabs defaultValue="basic" className="w-full">
            <TabsList className="grid grid-cols-2 w-full md:w-[400px]">
              <TabsTrigger value="basic">Basic</TabsTrigger>
              <TabsTrigger value="advanced">Advanced</TabsTrigger>
            </TabsList>
            
            <TabsContent value="basic" className="space-y-4 pt-4">
              <div className="space-y-4">
                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <Label htmlFor="dataset-select">Dataset</Label>
                    <TooltipWrapper text="Select a dataset to use for training">
                      <HelpCircle size={16} className="text-muted-foreground" />
                    </TooltipWrapper>
                  </div>
                  <Select
                    value={params.datasetId}
                    onValueChange={(value) => handleBasicParamChange("datasetId", value)}
                  >
                    <SelectTrigger>
                      <SelectValue placeholder={
                        datasets.length === 0 
                          ? "No datasets available" 
                          : "Select a dataset"
                      } />
                    </SelectTrigger>
                    <SelectContent>
                      {datasets.length === 0 ? (
                        <SelectItem value="none" disabled>
                          No datasets available
                        </SelectItem>
                      ) : (
                        datasets.map(dataset => (
                          <SelectItem key={dataset.id} value={dataset.id}>
                            {dataset.name}
                          </SelectItem>
                        ))
                      )}
                    </SelectContent>
                  </Select>
                  {datasets.length === 0 && (
                    <p className="text-xs text-amber-500 mt-1">
                      Please upload and validate a dataset first
                    </p>
                  )}
                </div>
                
                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <Label htmlFor="model-select">Model</Label>
                  </div>
                  <Select
                    value={params.model}
                    onValueChange={(value) => handleBasicParamChange("model", value)}
                  >
                    <SelectTrigger>
                      <SelectValue placeholder="Select a model" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="gemma-2b">Gemma 2B</SelectItem>
                      <SelectItem value="gemma-7b">Gemma 7B</SelectItem>
                      <SelectItem value="gemma-7b-it">Gemma 7B Instruction Tuned</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                
                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <Label htmlFor="epochs-slider">Epochs</Label>
                    <TooltipWrapper text="Number of complete passes through the dataset">
                      <HelpCircle size={16} className="text-muted-foreground" />
                    </TooltipWrapper>
                  </div>
                  <div className="flex items-center gap-4">
                    <Slider 
                      id="epochs-slider"
                      min={1} 
                      max={10} 
                      step={1}
                      value={[params.epochs]} 
                      onValueChange={(value) => handleBasicParamChange("epochs", value[0])}
                      className="flex-1" 
                    />
                    <div className="w-12 text-center">{params.epochs}</div>
                  </div>
                </div>
                
                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <Label htmlFor="lr-slider">Learning Rate</Label>
                    <TooltipWrapper text="Controls how quickly the model adapts to the problem">
                      <HelpCircle size={16} className="text-muted-foreground" />
                    </TooltipWrapper>
                  </div>
                  <div className="flex items-center gap-4">
                    <Slider 
                      id="lr-slider"
                      min={1e-6} 
                      max={1e-4} 
                      step={1e-6}
                      value={[params.learningRate]} 
                      onValueChange={(value) => handleBasicParamChange("learningRate", value[0])}
                      className="flex-1" 
                    />
                    <div className="w-20 text-center">{params.learningRate.toExponential(2)}</div>
                  </div>
                </div>
                
                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <Label htmlFor="bs-select">Batch Size</Label>
                    <TooltipWrapper text="Number of training examples used in one iteration">
                      <HelpCircle size={16} className="text-muted-foreground" />
                    </TooltipWrapper>
                  </div>
                  <Select
                    value={params.batchSize.toString()}
                    onValueChange={(value) => handleBasicParamChange("batchSize", parseInt(value))}
                  >
                    <SelectTrigger>
                      <SelectValue placeholder="Select batch size" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="1">1</SelectItem>
                      <SelectItem value="2">2</SelectItem>
                      <SelectItem value="4">4</SelectItem>
                      <SelectItem value="8">8</SelectItem>
                      <SelectItem value="16">16</SelectItem>
                      <SelectItem value="32">32</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                
                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <Label>Parameter-Efficient Fine-Tuning</Label>
                    <TooltipWrapper text="LoRA fine-tuning uses less memory and is faster">
                      <HelpCircle size={16} className="text-muted-foreground" />
                    </TooltipWrapper>
                  </div>
                  <div className="flex items-center space-x-2">
                    <Switch 
                      id="useLoRA" 
                      checked={params.useLoRA} 
                      onCheckedChange={(checked) => handleBasicParamChange("useLoRA", checked)} 
                    />
                    <Label htmlFor="useLoRA" className="cursor-pointer">Use LoRA</Label>
                  </div>
                </div>
              </div>
            </TabsContent>
            
            <TabsContent value="advanced" className="space-y-4 pt-4">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <Label htmlFor="max-len-slider">Max Length</Label>
                    <TooltipWrapper text="Maximum sequence length for inputs">
                      <HelpCircle size={16} className="text-muted-foreground" />
                    </TooltipWrapper>
                  </div>
                  <div className="flex items-center gap-4">
                    <Slider 
                      id="max-len-slider"
                      min={64} 
                      max={2048} 
                      step={64}
                      value={[params.maxLength]} 
                      onValueChange={(value) => handleBasicParamChange("maxLength", value[0])}
                      className="flex-1" 
                    />
                    <div className="w-16 text-center">{params.maxLength}</div>
                  </div>
                </div>
                
                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <Label htmlFor="warmup-steps">Warmup Steps</Label>
                    <TooltipWrapper text="Number of steps for learning rate warmup">
                      <HelpCircle size={16} className="text-muted-foreground" />
                    </TooltipWrapper>
                  </div>
                  <Input 
                    id="warmup-steps"
                    type="number" 
                    value={params.warmupSteps} 
                    onChange={(e) => handleBasicParamChange("warmupSteps", parseInt(e.target.value))}
                    min={0}
                  />
                </div>
                
                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <Label htmlFor="wd-slider">Weight Decay</Label>
                    <TooltipWrapper text="L2 regularization to prevent overfitting">
                      <HelpCircle size={16} className="text-muted-foreground" />
                    </TooltipWrapper>
                  </div>
                  <div className="flex items-center gap-4">
                    <Slider 
                      id="wd-slider"
                      min={0} 
                      max={0.1} 
                      step={0.001}
                      value={[params.weightDecay]} 
                      onValueChange={(value) => handleBasicParamChange("weightDecay", value[0])}
                      className="flex-1" 
                    />
                    <div className="w-16 text-center">{params.weightDecay.toFixed(3)}</div>
                  </div>
                </div>
                
                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <Label htmlFor="grad-accum">Gradient Accumulation Steps</Label>
                    <TooltipWrapper text="Simulate larger batch sizes with limited memory">
                      <HelpCircle size={16} className="text-muted-foreground" />
                    </TooltipWrapper>
                  </div>
                  <Select
                    value={params.gradientAccumulationSteps.toString()}
                    onValueChange={(value) => handleBasicParamChange("gradientAccumulationSteps", parseInt(value))}
                  >
                    <SelectTrigger>
                      <SelectValue placeholder="Select gradient accumulation steps" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="1">1</SelectItem>
                      <SelectItem value="2">2</SelectItem>
                      <SelectItem value="4">4</SelectItem>
                      <SelectItem value="8">8</SelectItem>
                      <SelectItem value="16">16</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                
                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <Label htmlFor="eval-steps">Evaluation Steps</Label>
                    <TooltipWrapper text="How often to evaluate the model during training">
                      <HelpCircle size={16} className="text-muted-foreground" />
                    </TooltipWrapper>
                  </div>
                  <Input 
                    id="eval-steps"
                    type="number" 
                    value={params.evalSteps} 
                    onChange={(e) => handleBasicParamChange("evalSteps", parseInt(e.target.value))}
                    min={10}
                  />
                </div>
                
                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <Label>Half Precision (FP16)</Label>
                    <TooltipWrapper text="Use 16-bit precision to reduce memory usage">
                      <HelpCircle size={16} className="text-muted-foreground" />
                    </TooltipWrapper>
                  </div>
                  <div className="flex items-center space-x-2">
                    <Switch 
                      id="useHalfPrecision" 
                      checked={params.useHalfPrecision} 
                      onCheckedChange={(checked) => handleBasicParamChange("useHalfPrecision", checked)} 
                    />
                    <Label htmlFor="useHalfPrecision" className="cursor-pointer">Enable</Label>
                  </div>
                </div>
                
                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <Label>Save Best Model</Label>
                    <TooltipWrapper text="Save model checkpoints with the best evaluation score">
                      <HelpCircle size={16} className="text-muted-foreground" />
                    </TooltipWrapper>
                  </div>
                  <div className="flex items-center space-x-2">
                    <Switch 
                      id="saveBestModel" 
                      checked={params.saveBestModel} 
                      onCheckedChange={(checked) => handleBasicParamChange("saveBestModel", checked)} 
                    />
                    <Label htmlFor="saveBestModel" className="cursor-pointer">Enable</Label>
                  </div>
                </div>
              </div>
            </TabsContent>
          </Tabs>
        </CardContent>
        <CardFooter className="flex justify-between">
          <Button variant="outline" onClick={handleResetToDefaults}>
            Reset to Defaults
          </Button>
          <Button 
            onClick={handleStartTraining} 
            className="btn-transition"
            disabled={datasets.length === 0}
          >
            <Zap size={16} className="mr-2" />
            Start Training
          </Button>
        </CardFooter>
      </Card>
    </TooltipProvider>
  );
}

function TooltipWrapper({ children, text }: { children: React.ReactNode; text: string }) {
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        {children}
      </TooltipTrigger>
      <TooltipContent>
        <p className="max-w-xs text-sm">{text}</p>
      </TooltipContent>
    </Tooltip>
  );
}
