
import { useState, useEffect } from 'react';
import { Link, useLocation } from 'react-router-dom';
import { cn } from '@/lib/utils';
import { 
  BarChart2, 
  Database, 
  Settings, 
  HardDrive, 
  Home,
  Menu,
  X
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { useIsMobile } from '@/hooks/use-mobile';
import { Gemma } from '@lobehub/icons';

interface NavItemProps {
  icon: React.ReactNode;
  label: string;
  to: string;
  isActive: boolean;
}

const navItems = [
  {
    icon: <Home size={20} />,
    label: 'Dashboard',
    to: '/dashboard',
  },
  {
    icon: <Database size={20} />,
    label: 'Datasets',
    to: '/datasets',
  },
  {
    icon: <BarChart2 size={20} />,
    label: 'Training',
    to: '/training',
  },
  {
    icon: <HardDrive size={20} />,
    label: 'Models',
    to: '/models',
  },
  {
    icon: <Settings size={20} />,
    label: 'Settings',
    to: '/settings',
  },
];

const NavItem = ({ icon, label, to, isActive }: NavItemProps) => {
  return (
    <Link 
      to={to} 
      className={cn(
        "flex items-center gap-3 px-4 py-3 rounded-lg transition-all duration-200",
        isActive 
          ? "bg-primary text-primary-foreground" 
          : "text-muted-foreground hover:bg-secondary hover:text-foreground"
      )}
    >
      {icon}
      <span className="font-medium">{label}</span>
    </Link>
  );
};

export function Navbar() {
  const location = useLocation();
  const isMobile = useIsMobile();
  const [isOpen, setIsOpen] = useState(false);
  
  // Close mobile menu when route changes
  useEffect(() => {
    setIsOpen(false);
  }, [location.pathname]);

  // Close mobile menu when window resizes above mobile breakpoint
  useEffect(() => {
    if (!isMobile) {
      setIsOpen(false);
    }
  }, [isMobile]);

  return (
    <header className="sticky top-0 z-40 w-full border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
      <div className="container flex items-center justify-between h-16 px-4 md:px-6">
      <div className="flex items-center gap-2">
  <Link to="/" className="flex items-center gap-2">
    {/* Replace the div with Gemma.Combine */}
    <Gemma.Combine size={45}/>
    <span className="font-bold text-xl tracking-tight">Studio</span>
  </Link>
</div>
        
        {isMobile ? (
          <Button 
            variant="ghost" 
            size="icon" 
            onClick={() => setIsOpen(!isOpen)}
            aria-label="Toggle menu"
          >
            {isOpen ? <X size={20} /> : <Menu size={20} />}
          </Button>
        ) : (
          <nav className="flex items-center gap-6">
            {navItems.map((item) => (
              <NavItem 
                key={item.label}
                icon={item.icon}
                label={item.label}
                to={item.to}
                isActive={location.pathname === item.to}
              />
            ))}
          </nav>
        )}
      </div>
      
      {/* Mobile Menu */}
      {isMobile && isOpen && (
        <div className="fixed inset-0 z-50 bg-background pt-16 animate-fade-in">
          <nav className="container flex flex-col gap-2 p-4">
            {navItems.map((item) => (
              <NavItem 
                key={item.label}
                icon={item.icon}
                label={item.label}
                to={item.to}
                isActive={location.pathname === item.to}
              />
            ))}
          </nav>
        </div>
      )}
    </header>
  );
}
