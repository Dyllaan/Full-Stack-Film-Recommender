import EditProfile from '../components/profile/EditProfile';
import { Link } from 'react-router-dom';
import ItemColumn from '../components/ItemColumn';
/**
 * 
 * The profile page.
 * shows recent notes and favourites, allows for editing of account information
 * @author Louis Figes
 * @generated GitHub Copilot was used in the creation of this code.
 */

function ProfilePage() {

  function favouriteItem(item, index) {
    return (
      <Link key={index} to={`/content/${item.content_id}`}>
        <div className="content-item rounded-lg" >
          <p className="text-md truncate">{item.title}</p>
        </div>
      </Link>
    );
  }

  function noteItem(note, index) {  
    return (
      <div className="content-item rounded-lg" key={index}>
        <p className="text-md truncate">{note.text}</p>
        <label className="text-xs text-slate-400">{note.created}</label>
        <Link to={`/content/${note.content_id}`}><p className="text-xs">Go to content</p></Link>
      </div>
    );
  }

  return (
    <div className="page w-[80vw] mx-auto">
      <div className="flex-row flex">
        <div className="w-[20vw]">
          <h4>Notes</h4>
          <ItemColumn endpoint={'notes/'} renderItem={noteItem} slice={5}/>
        </div>
        <EditProfile/>
      </div>
      <div className="overflow-hidden">
        <h4>Favourites</h4>
        <Link to="/profile/favourites"><span className="text-green-500">See all</span></Link>
        <div className="h-[15vh] overflow-y-scroll">
          <ItemColumn endpoint={'favourites/'} renderItem={favouriteItem} slice={5}/>
        </div>
      </div>
    </div>
  );
}

export default ProfilePage;
