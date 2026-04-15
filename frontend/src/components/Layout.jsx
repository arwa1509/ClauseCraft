import React from 'react';
import { Link, useLocation } from 'react-router-dom';
import { Scale, Search, FileText, BarChart2, ShieldCheck } from 'lucide-react';
import { clsx } from 'clsx';
import { twMerge } from 'tailwind-merge';

function cn(...inputs) {
  return twMerge(clsx(inputs));
}

const Layout = ({ children }) => {
  const location = useLocation();

  const navigation = [
    { name: 'Documents', href: '/', icon: FileText },
    { name: 'Legal Query', href: '/query', icon: Search },
    { name: 'Analytics', href: '/analytics', icon: BarChart2 },
  ];

  return (
    <div className="min-h-screen bg-legal-50 flex flex-col">
      {/* Navigation Header */}
      <header className="sticky top-0 z-50 glass border-b border-legal-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between h-16">
            <div className="flex items-center">
              <Link to="/" className="flex items-center space-x-2 group">
                <div className="bg-brand-600 p-1.5 rounded-lg group-hover:bg-brand-700 transition-colors">
                  <Scale className="h-6 w-6 text-white" />
                </div>
                <span className="text-xl font-bold text-legal-900 tracking-tight">
                  Lex<span className="text-brand-600">Analyze</span>
                </span>
              </Link>
              
              <nav className="hidden sm:ml-10 sm:flex sm:space-x-8">
                {navigation.map((item) => {
                  const isActive = location.pathname === item.href;
                  return (
                    <Link
                      key={item.name}
                      to={item.href}
                      className={cn(
                        "inline-flex items-center px-1 pt-1 border-b-2 text-sm font-medium transition-all duration-200",
                        isActive
                          ? "border-brand-600 text-legal-900"
                          : "border-transparent text-legal-500 hover:border-legal-300 hover:text-legal-700"
                      )}
                    >
                      <item.icon className="mr-2 h-4 w-4" />
                      {item.name}
                    </Link>
                  );
                })}
              </nav>
            </div>
            
            <div className="flex items-center space-x-4">
              <div className="hidden md:flex items-center space-x-2 bg-green-50 px-3 py-1 rounded-full border border-green-100">
                <ShieldCheck className="h-4 w-4 text-green-600" />
                <span className="text-xs font-medium text-green-700">Verifiable System</span>
              </div>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content Area */}
      <main className="flex-grow">
        <div className="max-w-7xl mx-auto py-8 px-4 sm:px-6 lg:px-8">
          {children}
        </div>
      </main>

      {/* Footer */}
      <footer className="bg-white border-t border-legal-200 py-8">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex flex-col md:flex-row justify-between items-center space-y-4 md:space-y-0 text-sm text-legal-500">
            <p>&copy; 2026 LexAnalyze AI. Production-Grade Legal NLP Pipeline.</p>
            <div className="flex space-x-6">
              <a href="#" className="hover:text-legal-900 transition-colors">API Documentation</a>
              <a href="#" className="hover:text-legal-900 transition-colors">Methodology</a>
              <a href="#" className="hover:text-legal-900 transition-colors">Github</a>
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
};

export default Layout;
