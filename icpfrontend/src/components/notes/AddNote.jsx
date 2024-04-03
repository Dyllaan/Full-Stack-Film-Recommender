import { useState, useEffect } from 'react';
import PropTypes from 'prop-types';
/**
 * 
 * Add a note to a piece of content
 * @author Louis Figes
 * @generated GitHub Copilot was used in the creation of this code.
 * 
 */
function AddNote(props) {
    const handleSubmit = props.handleSubmit;
    const [note, setNote] = useState("");
    const [isTouched, setIsTouched] = useState(false);
    const [error, setError] = useState(null);

    useEffect(() => {
        if (!isTouched) return;

        if (note.trim() === "") {
            setError("Note cannot be empty");
        } else if (note.length > 200) {
            setError("Note cannot be longer than 200 characters");
        } else {
            setError(null);
        }
    }, [note, isTouched]);

    const handleNoteChange = (e) => {
        setIsTouched(true);
        setNote(e.target.value);
    };

    const processSubmit = (e) => {
        e.preventDefault();
        if (!note.trim()) return;
        handleSubmit(note);
        setNote("");
        setIsTouched(false);
    };

    return (
        <div>
            <div className="flex gap-2">
                <textarea 
                    type="text" 
                    placeholder='Note' 
                    className={`${error && isTouched && "bg-red-900"}`}
                    value={note}
                    onChange={handleNoteChange}
                />
                <input
                    type="submit" 
                    value='Add Note' 
                    className='user-input w-fit'
                    onClick={processSubmit}
                    disabled={error || !isTouched}
                />
            </div>
            {error && isTouched && <p className="error">{error}</p>}
        </div>
    );
}

AddNote.propTypes = {
    handleSubmit: PropTypes.func.isRequired
};

export default AddNote;