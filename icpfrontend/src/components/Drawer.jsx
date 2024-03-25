import { useState } from 'react';
import Drawer from '@mui/material/Drawer';
import Button from '@mui/material/Button';

export default function MyDrawer() {
  const [isOpen, setIsOpen] = useState(false);

  const toggleDrawer = (open) => (event) => {
    if (event.type === 'keydown' && (event.key === 'Tab' || event.key === 'Shift')) {
      return;
    }

    setIsOpen(open);
  };

  return (
    <div>
      <Button onClick={toggleDrawer(true)}>Open Drawer</Button>
      <Drawer
        anchor='bottom' // Change to 'right', 'top', 'bottom' as needed
        open={isOpen}
        onClose={toggleDrawer(false)}
      >
        {/* Content here */}
        Your content goes here. Add list items, links, or any other content.
      </Drawer>
    </div>
  );
}