
import { WelcomeCard } from "@/components/dashboard/WelcomeCard";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { staggeredChildren, useAnimateOnMount } from "@/lib/animations";
import { Database, BarChart2, HardDrive, ArrowRight } from "lucide-react";
import { Link } from "react-router-dom";

const Dashboard = () => {
  const { styles } = useAnimateOnMount({
    type: 'fade',
    duration: 500
  });
  
  const getStaggerDelay = staggeredChildren(100);
  
  return (
    <div className="space-y-8" style={styles}>
      <h1 className="text-3xl font-bold tracking-tight">Dashboard</h1>
      
      <WelcomeCard />
      
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <DashboardCard 
          title="Datasets" 
          count={2}
          icon={<Database size={24} className="text-blue-500" />}
          linkTo="/datasets"
          animationDelay={getStaggerDelay(0).delay}
        />
        <DashboardCard 
          title="Training Jobs" 
          count={1}
          icon={<BarChart2 size={24} className="text-amber-500" />}
          status="In Progress"
          linkTo="/training"
          animationDelay={getStaggerDelay(1).delay}
        />
        <DashboardCard 
          title="Models" 
          count={1}
          icon={<HardDrive size={24} className="text-emerald-500" />}
          linkTo="/models"
          animationDelay={getStaggerDelay(2).delay}
        />
      </div>
      
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <Card className="card-hover" style={{ 
          ...useAnimateOnMount({ 
            type: 'slide', 
            direction: 'up', 
            delay: getStaggerDelay(3).delay 
          }).styles 
        }}>
          <CardHeader>
            <CardTitle className="text-lg">Recent Activity</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {[
                { time: "1 hour ago", action: "Training job started", model: "gemma-7b" },
                { time: "2 hours ago", action: "Dataset uploaded", model: "customer_support.csv" },
                { time: "1 day ago", action: "Model exported", model: "gemma-7b-ft-v1" },
              ].map((activity, index) => (
                <div key={index} className="flex items-start gap-3">
                  <div className="w-2 h-2 rounded-full bg-primary mt-1.5" />
                  <div>
                    <p className="font-medium">{activity.action}</p>
                    <div className="flex items-center gap-2 text-sm text-muted-foreground">
                      <span>{activity.time}</span>
                      <span>•</span>
                      <span>{activity.model}</span>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
        
        <Card className="card-hover" style={{ 
          ...useAnimateOnMount({ 
            type: 'slide', 
            direction: 'up', 
            delay: getStaggerDelay(4).delay 
          }).styles 
        }}>
          <CardHeader>
            <CardTitle className="text-lg">Quick Tips</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {[
                { title: "Prepare your dataset", description: "Ensure your dataset has clear examples of inputs and desired outputs" },
                { title: "Use parameter-efficient fine-tuning", description: "LoRA fine-tuning reduces memory usage and speeds up training" },
                { title: "Start with a small dataset", description: "Test your setup with a small dataset before scaling up" },
              ].map((tip, index) => (
                <div key={index} className="space-y-1">
                  <h3 className="font-medium">{tip.title}</h3>
                  <p className="text-sm text-muted-foreground">{tip.description}</p>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
};

interface DashboardCardProps {
  title: string;
  count: number;
  icon: React.ReactNode;
  status?: string;
  linkTo: string;
  animationDelay?: number;
}

const DashboardCard = ({ title, count, icon, status, linkTo, animationDelay = 0 }: DashboardCardProps) => {
  const { styles } = useAnimateOnMount({
    type: 'slide',
    direction: 'up',
    delay: animationDelay
  });
  
  return (
    <Link to={linkTo}>
      <Card className="card-hover h-full" style={styles}>
        <CardContent className="pt-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-muted-foreground">{title}</p>
              <h3 className="text-2xl font-bold mt-1">{count}</h3>
              {status && (
                <p className="text-xs mt-1 text-amber-500">{status}</p>
              )}
            </div>
            <div className="h-12 w-12 rounded-full bg-primary/10 flex items-center justify-center">
              {icon}
            </div>
          </div>
          <div className="flex items-center text-sm mt-4 text-primary link-hover">
            View All
            <ArrowRight size={14} className="ml-1" />
          </div>
        </CardContent>
      </Card>
    </Link>
  );
};

export default Dashboard;
