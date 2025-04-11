
import { ReactNode } from 'react';
import { Navbar } from './Navbar';
import { useLocation } from 'react-router-dom';
import { cn } from '@/lib/utils';

interface LayoutProps {
  children: ReactNode;
}

export function Layout({ children }: LayoutProps) {
  const location = useLocation();
  const isHomePage = location.pathname === '/';
  
  return (
    <div className="flex min-h-screen flex-col">
      {!isHomePage && <Navbar />}
      <main className={cn(
        "flex-1",
        isHomePage ? "px-0" : "container px-4 py-6 md:px-6 md:py-8"
      )}>
        {children}
      </main>
    </div>
  );
}
