import { useState, useRef } from 'react';
import NavItem from './NavItem';
import PropTypes from 'prop-types';
/**
 * This is the dropdown on the header
 * @author Louis Figes
 * @generated Github CoPilot and ChatGPT helped with this code, INCLUDING THE ARROW THAT ROTATES
 * I couldnt get the arrow working
 * 
 */

function UserDropdown(props) {
  const [isOpen, setIsOpen] = useState(false);
  const closingTimer = useRef();

  const { user } = props;

  const openDropdown = () => {
    clearTimeout(closingTimer.current);
    setIsOpen(true);
  };

  const startClosingDropdown = () => {
    closingTimer.current = setTimeout(() => {
      setIsOpen(false);
    }, 300);
  };

  const cancelClosingDropdown = () => {
    clearTimeout(closingTimer.current);
  };

  return (
    <div className="relative inline-block text-left">
      <div onMouseEnter={openDropdown} onMouseLeave={startClosingDropdown}>
        <button type="button" className="nav-item">
          {user.username}
          <svg className={`-mr-1 ml-2 h-5 w-5 transform transition-transform ${isOpen ? 'rotate-180' : ''}`} xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="white" aria-hidden="true">
            <path fillRule="evenodd" d="M5.293 7.293a1 1 0 011.414 0L10 10.586l3.293-3.293a1 1 0 011.414 1.414l-4 4a1 1 0 01-1.414 0l-4-4a1 1 0 010-1.414z" clipRule="evenodd" />
          </svg>
        </button>
      </div>

      {isOpen && (
        <div className="absolute right-0 z-10 mt-2 w-56 origin-top-right rounded-f primary" onMouseEnter={cancelClosingDropdown} onMouseLeave={startClosingDropdown}>
          <div className="py-1">
            <div className="">
              <NavItem pageName="Your profile" pageUrl="/profile" />
            </div>
            <div className="italic">
              <NavItem pageName="Sign Out" pageUrl="/logout" />
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

UserDropdown.propTypes = {
  user: PropTypes.object.isRequired,
};

export default UserDropdown;
