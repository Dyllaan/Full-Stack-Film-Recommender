import PropTypes from "prop-types";
import { useEffect } from "react";

/**
 * Message component, handles error and success
 * messages.
 * @author Louis Figes
 * @generated GitHub Copilot was used in the creation of this code.
 */
function Message({ message, clearMessage, type }) {
    useEffect(() => {
        const timer = setTimeout(() => {
            if (clearMessage) {
                clearMessage();
            }
        }, 4000);
        return () => clearTimeout(timer);
    }, [message, clearMessage]);

    const messageColor = type === 'success' ? 'text-green-400' : 'text-red-400';

    return (
        <div className={`flex pr-4 items-center gap-2 ${messageColor}`}>
            <p className={`${messageColor} text-lg`}>{message}</p>
        </div>
    );
}

Message.propTypes = {
    message: PropTypes.string.isRequired,
    clearMessage: PropTypes.func,
    type: PropTypes.oneOf(['success', 'error']).isRequired
};

export default Message;
