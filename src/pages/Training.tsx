import { HyperparameterConfig } from "@/components/training/HyperparameterConfig";
import { TrainingProgress } from "@/components/training/TrainingProgress";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { useState, useEffect } from "react";
import { useLocation } from "react-router-dom";
import { getLatestTrainingJob } from "@/services/trainingService";

const Training = () => {
  const location = useLocation();
  const [activeTab, setActiveTab] = useState(location.state?.activeTab || "config");
  
  useEffect(() => {
    if (location.state?.activeTab === "progress") {
      setActiveTab("progress");
    } else {
      const latestJob = getLatestTrainingJob();
      if (latestJob && (latestJob.status === "training" || latestJob.status === "paused")) {
        setActiveTab("progress");
      }
    }
  }, [location.state]);
  
  return (
    <div className="space-y-6">
      <h1 className="text-3xl font-bold tracking-tight">Training</h1>
      
      <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-4">
        <TabsList className="grid w-full grid-cols-2 md:w-[400px]">
          <TabsTrigger value="config">Configuration</TabsTrigger>
          <TabsTrigger value="progress">Progress</TabsTrigger>
        </TabsList>
        
        <TabsContent value="config" className="space-y-4">
          <HyperparameterConfig />
        </TabsContent>
        
        <TabsContent value="progress" className="space-y-4">
          <TrainingProgress />
        </TabsContent>
      </Tabs>
    </div>
  );
};

export default Training;
