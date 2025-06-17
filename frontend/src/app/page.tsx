'use client';
import { ChatLayout } from "@/components/chat/chat-layout";
import React, { useEffect, useCallback, useState } from 'react';

export default function Home() {
  const [selectedConversationId, setSelectedConversationId] = useState<string | null>(null);
  const [resetTrigger, setResetTrigger] = useState<number>(0); // Add a reset trigger to force reset
  
  // Check URL for conversation parameter on initial load and popstate events
  useEffect(() => {
    const handleConversationFromURL = () => {
      // Read conversation ID from URL if present
      const urlParams = new URLSearchParams(window.location.search);
      const conversationId = urlParams.get('conversation');
      setSelectedConversationId(conversationId);
    };
    
    // Parse URL on initial load
    handleConversationFromURL();
    
    // Listen for navigation events (back/forward buttons or history.pushState)
    window.addEventListener('popstate', handleConversationFromURL);
    
    return () => {
      window.removeEventListener('popstate', handleConversationFromURL);
    };
  }, []);
  
  // Use useCallback to ensure the function is stable between renders
  const onNewChat = useCallback(() => {
    // Force a complete reset of the chat state
    setSelectedConversationId(null);
    
    // Add a reset trigger increment to force component reset
    setResetTrigger(prev => prev + 1);
    
    // Update the URL to remove the conversation parameter
    window.history.pushState({}, '', '/');
    
    // Additional debugging if needed
    console.log('Starting new chat, reset triggered');
  }, []);
  
  return (
    <main className="min-h-screen">
      <ChatLayout 
        key={`chat-${resetTrigger}`} // Force remount on reset
        selectedConversationId={selectedConversationId} 
        onNewChat={onNewChat} 
      />
    </main>
  );
}
