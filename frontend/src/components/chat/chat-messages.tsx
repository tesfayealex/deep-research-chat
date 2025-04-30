import React from 'react';
import { ScrollArea } from "@/components/ui/scroll-area";
import MessageBubble from './message-bubble';

interface Message {
  id: string; // Add unique id for key prop
  sender: 'user' | 'agent';
  text: string;
}

interface ChatMessagesProps {
  messages: Message[];
}

export default function ChatMessages({ messages }: ChatMessagesProps) {
  return (
    <ScrollArea className="flex-1 p-4">
      <div className="flex flex-col gap-4">
        {messages.map((msg) => (
          <MessageBubble key={msg.id} message={msg} />
        ))}
        {/* TODO: Add ref and scroll-to-bottom logic */}
      </div>
    </ScrollArea>
  );
} 