import { PropTypes } from "prop-types";
import AuthorContainer from "./AuthorContainer";
import Notes from "./notes/Notes";
import useAuth from "../auth/useAuth";
import RestrictedElement from "../profile/RestrictedElement";
import useFetchData from '../../hooks/useFetchData';
import { useEffect, useState } from "react";
import PreviewThumbnail from "../preview/PreviewThumbnail";
import { Link } from "react-router-dom";
import FavouriteButton from "./FavouriteButton";
/**
 * 
 * @author Louis Figes 
 * @generated GitHub Copilot was used in the creation of this code.
 */
function SelectedContent(props) {
    const contentId = props.contentId;
    const [selectedContent, setSelectedContent] = useState(null);
    const { signedIn, addNote, deleteNote, isFavourited, addFavourite, deleteFavourite } = useAuth();

    const { data: authorsAndAffiliations, setEndpoint: setAuthorEndpoint, reloadData: reloadAuthorData } = useFetchData(`author-and-affiliation/?content=${contentId}`);
    const { data: contentData, setEndpoint: setContentEndpoint, reloadData, loading, showInfo } = useFetchData(`content/?id=${contentId}`);

    useEffect(() => {
        setContentEndpoint(`content/?id=${contentId}`);
        reloadData();
        setAuthorEndpoint(`author-and-affiliation/?content=${contentId}`);
        reloadAuthorData();
    }, [contentId, setContentEndpoint]);
    
    
    useEffect(() => {
        if (contentData && contentData.length > 0) {
            setSelectedContent(contentData[0]);
        }
    }, [contentData]);


    if (!selectedContent) {
        return;
    }

    if(!loading && !selectedContent) {
        return <div>Content not found</div>;
    }

    function getDOI() {
        return selectedContent.doi_link.replace("https://doi.org/", "");
    }

    return (
        <div className="overflow-x-auto p-4 rounded-lg mx-auto px-4 overflow-y-scroll text-center">
            {showInfo()}
            <div>
                <div className="xl:grid xl:grid-cols-10 overflow-hidden outline outline-1 outline-slate-500">
                    <div className="col-span-8 secondary px-2">
                        <h2>{selectedContent.title}</h2>
                    </div>
                    <div className="flex-row flex xl:flex-col xl:gap-0 gap-2 col-span-2 break-words bg-slate-600 px-2 items-center">
                    {signedIn &&<FavouriteButton signedIn={signedIn} isFavourited={isFavourited} selectedContent={selectedContent} contentId={contentId} addFavourite={addFavourite} deleteFavourite={deleteFavourite} />}
                        <h4 className="text-red-300">{selectedContent.type}</h4>
                        <Link to={selectedContent.doi_link} className="hover:text-blue-400">{getDOI()}</Link>
                        {selectedContent.award && <p className="award">{selectedContent.award} awards</p>}
                    </div>
                </div>
                <div className="flex flex-col lg:flex-row justify-between my-4 gap-4">
                    {selectedContent.preview_video && 
                    <div className="w-full lg:w-[20vw]">
                        <Link to={selectedContent.preview_video} target='_blank'>
                            <PreviewThumbnail title={selectedContent.title} previewVideo={selectedContent.preview_video} />
                        </Link>
                    </div>}
                    <div>
                        {selectedContent.abstract && <p className="text-slate-100">{selectedContent.abstract}</p>}
                    </div>
                </div>
            </div>
            <div className="flex justify-center text-center items-center mt-4">
                <AuthorContainer authorData={authorsAndAffiliations} />
            </div>
            <div className="flex">
                {signedIn &&
                    <Notes contentId={contentId} content={selectedContent} addNote={addNote} deleteNote={deleteNote} />
                }
            </div>
            
        {!signedIn && <RestrictedElement text="add notes" />}
        </div>
    );
}

SelectedContent.propTypes = {
    contentId: PropTypes.number.isRequired,
};

export default SelectedContent;
