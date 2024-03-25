import PropTypes from 'prop-types';
import AddNote from './AddNote';
import useFetchData from '../../../hooks/useFetchData';
import { useEffect, useState } from 'react';
import Search from '../../search/Search';
import { Link } from 'react-router-dom';
/**
 * @author Louis Figes
 * Notes component shows the notes for a specific piece of content, is used in the SelectedContent component
 * @generated Github Copilot was used during the creation of this component
 */

function Notes(props) {
    const { addNote, deleteNote } = props;
    const contentId = props.contentId;
    const {data : notes, reloadData, setEndpoint, loading} = useFetchData(`notes/?content_id=${contentId}`);
    const [search, setSearch] = useState('');

    useEffect(() => {
        if(!loading ) return;
        reloadData();
    }, [loading]);

    useEffect(() => {
        setEndpoint(`notes/?content_id=${contentId}`);
        reloadData();
    }, [contentId]);
    

    const filteredNotes = notes ? notes.filter(note =>
        note.text.toLowerCase().includes(search.toLowerCase())
      ) : [];

    const handleSearch = (e) => {
        setSearch(e.target.value);
    }

    async function handleSubmit(note) {
        if (note.trim() === "") {
            return;
        }
        const response = await addNote(note, contentId);
        if(response) {
            reloadData();
        }
    }

    const handleDelete = async (id) => {
        console.log(id);
        const response = await deleteNote(id);
        if(response) {
            reloadData();
        }
    }

    return (
        <div className="flex flex-col p-4 gap-2 mx-auto">
            <div className="flex flex-col gap-2 max-w-xl mx-auto">
                <AddNote handleSubmit={handleSubmit} />
                <Search handleSearchChange={handleSearch} placeHolder="Search Notes"/>
            </div>
            {filteredNotes.length > 0 && (
                <div className="grid grid-cols-2 lg:grid-cols-3 2xl:grid-cols-4 gap-2 mx-2">
                    {filteredNotes.sort((a, b) => new Date(b.created) - new Date(a.created)).map((note, index) => (
                        <div className="primary flex flex-row rounded-lg p-2" key={index}>
                            <img src="/x-icon.svg" alt="Delete" onClick={() => handleDelete(note.note_id)} className="cursor-pointer w-6 h-6 md:w-5 md:h-5"/>
                            <div className="flex-grow overflow-hidden">
                                <p className="word-wrap break-words">{note.text}</p>
                                <p className="text-red-200 text-xs">{note.created}</p>
                            </div>
                        </div>
                    ))}
                </div>
            )}
            {filteredNotes.length === 0 && search && (
                <div className="flex flex-col items-center justify-center h-full">
                    <p className="text-center text-gray-400">No notes found</p>
                </div>
            )}
            <div>
                <Link className="text-xs" to="https://www.freeiconspng.com/img/13612">Free Close Icon</Link>
            </div>
        </div>
    );
}

Notes.propTypes = {
    contentId: PropTypes.number.isRequired,
    addNote: PropTypes.func.isRequired,
    deleteNote: PropTypes.func.isRequired,
};

export default Notes;
