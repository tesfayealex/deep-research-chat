'use client';

import React, { useEffect, useState } from 'react';
import { Plus, MessageSquare, X } from 'lucide-react';
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";

interface Conversation {
  id: string;
  title: string;
  timestamp: string;
}

interface SidebarProps {
  selectedConversationId: string | null;
  onSelectConversation: (id: string) => void;
  onNewChat: () => void;
}

export function Sidebar({ 
  selectedConversationId, 
  onSelectConversation, 
  onNewChat 
}: SidebarProps) {
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [loading, setLoading] = useState(false);
  const [isOpen, setIsOpen] = useState(true);
  const backendUrl = process.env.NEXT_PUBLIC_BACKEND_API_URL || 'http://localhost:8000';

  // Load conversations from backend
  useEffect(() => {
    const fetchConversations = async () => {
      setLoading(true);
      try {
        const response = await fetch(`${backendUrl}/api/history`);
        if (!response.ok) {
          throw new Error(`Failed to fetch conversations: ${response.status}`);
        }
        const data = await response.json();
        // Ensure we're setting the actual array, not the whole object
        setConversations(Array.isArray(data.conversations) ? data.conversations : []);
      } catch (error) {
        console.error("Error loading conversations:", error);
        setConversations([]);  // Set to empty array on error
      } finally {
        setLoading(false);
      }
    };

    fetchConversations();
  }, [backendUrl]);

  const formatDate = (dateString: string) => {
    const date = new Date(dateString);
    return new Intl.DateTimeFormat('en-US', { 
      month: 'short', 
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    }).format(date);
  };

  // For mobile responsiveness
  const toggleSidebar = () => {
    setIsOpen(!isOpen);
  };

  return (
    <>
      {/* Mobile toggle button */}
      <Button 
        variant="outline" 
        size="icon" 
        className="absolute top-4 left-4 z-50 md:hidden"
        onClick={toggleSidebar}
      >
        {isOpen ? <X className="h-4 w-4" /> : <MessageSquare className="h-4 w-4" />}
      </Button>
      
      {/* Sidebar */}
      <div className={`${isOpen ? 'translate-x-0' : '-translate-x-full'} 
                      transform transition-transform duration-200 ease-in-out
                      w-64 h-full border-r bg-background flex flex-col
                      md:translate-x-0 fixed md:relative z-40`}>
        {/* New Chat Button */}
        <div className="p-4 border-b">
          <Button 
            variant="default" 
            className="w-full justify-start text-left"
            onClick={onNewChat}
          >
            <Plus className="mr-2 h-4 w-4" />
            New Chat
          </Button>
        </div>
        
        {/* Chat History */}
        <ScrollArea className="flex-1 p-2">
          {loading ? (
            <div className="text-center py-4 text-sm text-muted-foreground">Loading conversations...</div>
          ) : conversations.length === 0 ? (
            <div className="text-center py-4 text-sm text-muted-foreground">No conversations yet</div>
          ) : (
            <div className="space-y-1">
              {conversations.map((convo) => (
                <Button
                  key={convo.id}
                  variant={selectedConversationId === convo.id ? "secondary" : "ghost"}
                  className="w-full justify-start text-left h-auto py-2 px-3"
                  onClick={() => onSelectConversation(convo.id)}
                >
                  <div className="flex flex-col">
                    <span className="line-clamp-1 font-medium">{convo.title}</span>
                    <span className="text-xs text-muted-foreground">
                      {formatDate(convo.timestamp)}
                    </span>
                  </div>
                </Button>
              ))}
            </div>
          )}
        </ScrollArea>
      </div>
    </>
  );
} 