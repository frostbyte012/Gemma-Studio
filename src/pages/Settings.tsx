
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Switch } from "@/components/ui/switch";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Input } from "@/components/ui/input";
import { useAnimateOnMount } from "@/lib/animations";
import { toast } from "sonner";
import { HardDrive, Cloud, Download, RefreshCw, Save } from "lucide-react";

const Settings = () => {
  const { styles } = useAnimateOnMount({
    type: 'fade',
    duration: 500
  });
  
  const handleSaveSettings = () => {
    toast.success("Settings saved successfully");
  };
  
  return (
    <div className="space-y-6" style={styles}>
      <h1 className="text-3xl font-bold tracking-tight">Settings</h1>
      
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <Card className="border shadow-sm">
          <CardHeader>
            <CardTitle>General Settings</CardTitle>
            <CardDescription>Configure general application settings</CardDescription>
          </CardHeader>
          <CardContent className="space-y-6">
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <div className="space-y-0.5">
                  <Label htmlFor="dark-mode">Dark Mode</Label>
                  <p className="text-sm text-muted-foreground">
                    Enable dark mode for the application
                  </p>
                </div>
                <Switch id="dark-mode" defaultChecked />
              </div>
              
              <div className="flex items-center justify-between">
                <div className="space-y-0.5">
                  <Label htmlFor="notifications">Notifications</Label>
                  <p className="text-sm text-muted-foreground">
                    Receive notifications about training jobs
                  </p>
                </div>
                <Switch id="notifications" defaultChecked />
              </div>
              
              <div className="space-y-2">
                <Label htmlFor="project-name">Project Name</Label>
                <Input id="project-name" defaultValue="Gemma Fine-tuning Project" />
              </div>
            </div>
          </CardContent>
        </Card>
        
        <Card className="border shadow-sm">
          <CardHeader>
            <CardTitle>Storage Settings</CardTitle>
            <CardDescription>Configure where to store models and datasets</CardDescription>
          </CardHeader>
          <CardContent className="space-y-6">
            <div className="space-y-4">
              <div className="flex items-center gap-4 p-3 border rounded-lg">
                <div className="h-10 w-10 rounded-full bg-primary/10 flex items-center justify-center">
                  <HardDrive className="h-5 w-5 text-primary" />
                </div>
                <div className="flex-1">
                  <p className="font-medium">Local Storage</p>
                  <p className="text-sm text-muted-foreground">Store models and datasets locally</p>
                </div>
                <Switch id="local-storage" defaultChecked />
              </div>
              
              <div className="flex items-center gap-4 p-3 border rounded-lg">
                <div className="h-10 w-10 rounded-full bg-primary/10 flex items-center justify-center">
                  <Cloud className="h-5 w-5 text-primary" />
                </div>
                <div className="flex-1">
                  <p className="font-medium">Cloud Storage</p>
                  <p className="text-sm text-muted-foreground">Store models and datasets in the cloud</p>
                </div>
                <Switch id="cloud-storage" />
              </div>
            </div>
          </CardContent>
        </Card>
        
        <Card className="border shadow-sm">
          <CardHeader>
            <CardTitle>Model Defaults</CardTitle>
            <CardDescription>Configure default settings for model training</CardDescription>
          </CardHeader>
          <CardContent className="space-y-6">
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <div className="space-y-0.5">
                  <Label htmlFor="auto-save">Auto-save Checkpoints</Label>
                  <p className="text-sm text-muted-foreground">
                    Automatically save model checkpoints during training
                  </p>
                </div>
                <Switch id="auto-save" defaultChecked />
              </div>
              
              <div className="space-y-2">
                <Label htmlFor="checkpoint-frequency">Checkpoint Frequency (steps)</Label>
                <Input id="checkpoint-frequency" type="number" defaultValue="100" />
              </div>
              
              <div className="flex items-center justify-between">
                <div className="space-y-0.5">
                  <Label htmlFor="mixed-precision">Mixed Precision Training</Label>
                  <p className="text-sm text-muted-foreground">
                    Use FP16/BF16 precision to speed up training
                  </p>
                </div>
                <Switch id="mixed-precision" defaultChecked />
              </div>
            </div>
          </CardContent>
        </Card>
        
        <Card className="border shadow-sm">
          <CardHeader>
            <CardTitle>Export & Backup</CardTitle>
            <CardDescription>Configure export and backup settings</CardDescription>
          </CardHeader>
          <CardContent className="space-y-6">
            <div className="space-y-4">
              <Button variant="outline" className="w-full justify-start">
                <Download className="mr-2 h-4 w-4" />
                Export All Settings
              </Button>
              
              <Button variant="outline" className="w-full justify-start">
                <RefreshCw className="mr-2 h-4 w-4" />
                Restore Default Settings
              </Button>
              
              <Button variant="outline" className="w-full justify-start">
                <Save className="mr-2 h-4 w-4" />
                Backup Projects
              </Button>
            </div>
          </CardContent>
        </Card>
      </div>
      
      <div className="flex justify-end">
        <Button onClick={handleSaveSettings}>Save Settings</Button>
      </div>
    </div>
  );
};

export default Settings;
