
import { useState, useEffect } from "react";
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Pause, Play, RefreshCw, AlertCircle, StopCircle, Terminal } from "lucide-react";
import { useAnimateOnMount } from "@/lib/animations";
import { toast } from "sonner";
import { useLocation } from "react-router-dom";
import {
  Line,
  LineChart,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip as RechartsTooltip,
  ResponsiveContainer,
} from "recharts";
import { 
  getLatestTrainingJob, 
  getTrainingJob, 
  pauseTraining, 
  resumeTraining, 
  stopTraining,
  formatTime 
} from "@/services/trainingService";
import { TrainingState } from "@/services/trainingService";

export function TrainingProgress() {
  const [trainingJob, setTrainingJob] = useState<TrainingState | null>(null);
  const location = useLocation();
  
  const { styles } = useAnimateOnMount({
    type: 'slide',
    direction: 'up',
    duration: 500
  });
  
  // Initialize with job from route state or get latest
  useEffect(() => {
    const jobId = location.state?.jobId;
    
    if (jobId) {
      const job = getTrainingJob(jobId);
      if (job) {
        setTrainingJob(job);
        return;
      }
    }
    
    // If no jobId in route or job not found, get latest
    const latestJob = getLatestTrainingJob();
    if (latestJob) {
      setTrainingJob(latestJob);
    }
  }, [location.state?.jobId]);
  
  // Poll for training job updates
  useEffect(() => {
    if (!trainingJob) return;
    
    const intervalId = setInterval(() => {
      const updatedJob = getTrainingJob(trainingJob.id);
      if (updatedJob) {
        setTrainingJob(updatedJob);
      }
    }, 1000);
    
    return () => clearInterval(intervalId);
  }, [trainingJob]);
  
  const togglePause = () => {
    if (!trainingJob) return;
    
    if (trainingJob.status === 'training') {
      const updatedJob = pauseTraining(trainingJob.id);
      if (updatedJob) {
        setTrainingJob(updatedJob);
      }
    } else if (trainingJob.status === 'paused') {
      const updatedJob = resumeTraining(trainingJob.id);
      if (updatedJob) {
        setTrainingJob(updatedJob);
      }
    }
  };
  
  const handleStop = () => {
    if (!trainingJob) return;
    
    const updatedJob = stopTraining(trainingJob.id);
    if (updatedJob) {
      setTrainingJob(updatedJob);
    }
  };
  
  // If no training job is available yet
  if (!trainingJob) {
    return (
      <Card className="border shadow-sm" style={styles}>
        <CardHeader>
          <CardTitle>Training Progress</CardTitle>
          <CardDescription>
            No active training job found. Configure and start a new training.
          </CardDescription>
        </CardHeader>
        <CardContent className="text-center py-12">
          <div className="space-y-4">
            <AlertCircle size={48} className="mx-auto text-muted-foreground" />
            <p className="text-muted-foreground">
              No training in progress. Go to the Configuration tab to start a new training job.
            </p>
          </div>
        </CardContent>
        <CardFooter className="justify-center">
          <Button variant="outline" onClick={() => window.history.back()}>
            Back to Configuration
          </Button>
        </CardFooter>
      </Card>
    );
  }
  
  const isPaused = trainingJob.status === 'paused';
  const isCompleted = trainingJob.status === 'completed';
  const isTraining = trainingJob.status === 'training';
  const isPreparing = trainingJob.status === 'preparing';
  
  return (
    <Card className="border shadow-sm" style={styles}>
      <CardHeader>
        <div className="flex items-center justify-between">
          <div>
            <CardTitle>Training Progress</CardTitle>
            <CardDescription>
              Model: {trainingJob.params.model} • Dataset: {trainingJob.params.datasetId}
            </CardDescription>
          </div>
          <Badge 
            variant={isPaused ? "outline" : "default"} 
            className={
              isPaused ? "bg-amber-100 dark:bg-amber-950 text-amber-800 dark:text-amber-400" : 
              isCompleted ? "bg-green-100 dark:bg-green-950 text-green-800 dark:text-green-400" :
              isPreparing ? "bg-blue-100 dark:bg-blue-950 text-blue-800 dark:text-blue-400" :
              "animate-pulse"
            }
          >
            {isPaused ? "Paused" : isCompleted ? "Completed" : isPreparing ? "Preparing" : "Training"}
          </Badge>
        </div>
      </CardHeader>
      <CardContent className="space-y-6">
        <div className="space-y-2">
          <div className="flex justify-between text-sm">
            <span>Progress: {trainingJob.currentStep}/{trainingJob.totalSteps} steps</span>
            <span>{Math.round(trainingJob.progress)}%</span>
          </div>
          <Progress value={trainingJob.progress} className="h-2" />
          <div className="flex justify-between text-xs text-muted-foreground">
            <span>Elapsed: {formatTime(trainingJob.elapsedTime)}</span>
            <span>ETA: {formatTime(trainingJob.estimatedTimeRemaining)}</span>
          </div>
        </div>
        
        <Tabs defaultValue="metrics" className="w-full">
          <TabsList className="grid grid-cols-3 w-full">
            <TabsTrigger value="metrics">Metrics</TabsTrigger>
            <TabsTrigger value="evaluation">Evaluation</TabsTrigger>
            <TabsTrigger value="logs">Logs</TabsTrigger>
          </TabsList>
          
          <TabsContent value="metrics" className="space-y-4 pt-4">
            <div className="h-[300px] w-full">
              {trainingJob.metrics.length > 0 ? (
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart
                    data={trainingJob.metrics}
                    margin={{ top: 10, right: 10, left: 0, bottom: 0 }}
                  >
                    <CartesianGrid strokeDasharray="3 3" opacity={0.2} />
                    <XAxis 
                      dataKey="step" 
                      label={{ value: 'Steps', position: 'insideBottomRight', offset: -5 }} 
                    />
                    <YAxis label={{ value: 'Loss', angle: -90, position: 'insideLeft' }} />
                    <RechartsTooltip 
                      formatter={(value: number) => [value.toFixed(4), '']}
                      labelFormatter={(label) => `Step ${label}`}
                    />
                    <Line 
                      type="monotone" 
                      dataKey="trainingLoss" 
                      name="Training Loss"
                      stroke="hsl(var(--primary))" 
                      strokeWidth={2}
                      dot={false}
                      activeDot={{ r: 6 }}
                    />
                    <Line 
                      type="monotone" 
                      dataKey="validationLoss" 
                      name="Validation Loss"
                      stroke="hsl(var(--destructive))" 
                      strokeWidth={2} 
                      dot={false}
                      activeDot={{ r: 6 }}
                    />
                  </LineChart>
                </ResponsiveContainer>
              ) : (
                <div className="h-full flex items-center justify-center">
                  <p className="text-muted-foreground">
                    Waiting for training metrics...
                  </p>
                </div>
              )}
            </div>
          </TabsContent>
          
          <TabsContent value="evaluation" className="space-y-4 pt-4">
            <div className="border rounded-lg overflow-hidden">
              <table className="w-full">
                <thead>
                  <tr className="bg-secondary">
                    <th className="px-4 py-3 text-left font-medium">Metric</th>
                    <th className="px-4 py-3 text-left font-medium">Value</th>
                    <th className="px-4 py-3 text-left font-medium">Change</th>
                  </tr>
                </thead>
                <tbody className="divide-y">
                  {trainingJob.evaluationMetrics.map((item, index) => {
                    const change = parseFloat((item.value - item.previous).toFixed(2));
                    const isPositive = change > 0;
                    
                    return (
                      <tr key={index} className="hover:bg-secondary/50">
                        <td className="px-4 py-3">{item.metric}</td>
                        <td className="px-4 py-3 font-medium">{item.value.toFixed(2)}</td>
                        <td className="px-4 py-3">
                          <span className={isPositive ? "text-emerald-500" : "text-red-500"}>
                            {isPositive ? "+" : ""}{change.toFixed(2)}
                          </span>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
            
            <div className="border rounded-lg p-4 bg-muted/30">
              <h4 className="font-medium mb-2">Evaluation Examples</h4>
              <div className="space-y-3">
                <div className="space-y-1">
                  <p className="text-sm font-medium">Input:</p>
                  <p className="text-sm bg-muted p-2 rounded">How can I reset my password if I've forgotten it?</p>
                </div>
                <div className="space-y-1">
                  <p className="text-sm font-medium">Model Output:</p>
                  <p className="text-sm bg-muted p-2 rounded">To reset your forgotten password, go to the login page and click on "Forgot Password". You'll receive an email with a link to create a new password. If you don't receive the email, check your spam folder or contact our support team for assistance.</p>
                </div>
              </div>
            </div>
          </TabsContent>
          
          <TabsContent value="logs" className="pt-4">
            <div className="bg-black text-green-400 font-mono text-sm p-4 rounded-lg h-[300px] overflow-y-auto">
              {trainingJob.logs.map((line, index) => (
                <div key={index} className="pb-1">{line}</div>
              ))}
              {isTraining && (
                <div className="animate-pulse">▋</div>
              )}
            </div>
          </TabsContent>
        </Tabs>
      </CardContent>
      <CardFooter className="flex justify-between">
        <div className="flex items-center gap-2">
          <Button 
            variant="outline" 
            onClick={togglePause} 
            className="space-x-1"
            disabled={isCompleted || isPreparing}
          >
            {isPaused ? (
              <>
                <Play size={16} />
                <span>Resume</span>
              </>
            ) : (
              <>
                <Pause size={16} />
                <span>Pause</span>
              </>
            )}
          </Button>
          <Button 
            variant="outline" 
            onClick={handleStop} 
            className="space-x-1"
            disabled={isCompleted || isPreparing}
          >
            <StopCircle size={16} />
            <span>Stop</span>
          </Button>
        </div>
        <div className="flex items-center gap-2">
          <Button 
            variant="outline" 
            className="space-x-1"
            disabled={!isCompleted}
            onClick={() => window.location.href = "/models"}
          >
            <Terminal size={16} />
            <span>Export Model</span>
          </Button>
        </div>
      </CardFooter>
    </Card>
  );
}
